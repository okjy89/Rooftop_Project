import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ultralytics import YOLO

from MLPNetworkV2 import KernelPredictor  # 기존에 정의한 클래스

class MLPTrainer:
    def __init__(self, backbone, feature_channels,
                 yolo_weights='master_yolov8.pt',
                 lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) Backbone (parse_model 등)
        self.model = backbone.to(self.device)

        # 2) KernelPredictor
        self.kernel_pred = KernelPredictor(feature_channels, kernel_size=3).to(self.device)

        # 3) Master YOLOv8 (weight freeze)
        self.yolo = YOLO(yolo_weights)            # Ultralytics YOLO 객체
        self.yolo.model.to(self.device)           # 내부 nn.Module을 device로 이동
        # YOLO 파라미터는 모두 freeze
        for p in self.yolo.model.parameters():
            p.requires_grad = False
        # eval 모드로 두면 batchnorm, dropout 등 inference 환경에 맞춰짐. 
        # (물론 gradient를 여기서 다시 살려줄 것이므로, eval 모드여도 raw logits는 얻을 수 있습니다.)
        self.yolo.model.eval()

        # 4) Optimizer: backbone + kernel predictor만 업데이트
        params = list(self.model.parameters()) + list(self.kernel_pred.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    def train_step(self, images, _targets=None):
        """
        images: [B, 3, H, W]
        _targets: unused — YOLO confidence를 직접 loss로 쓰므로
        """
        # 1) Backbone 및 KernelPredictor를 train 모드
        self.model.train()
        self.kernel_pred.train()
        # YOLO는 eval 모드로 고정 (파라미터는 freeze 상태, BatchNorm 등도 eval)
        self.yolo.model.eval()

        imgs = images.to(self.device)  # [B, 3, H, W]

        # 2) feature map 추출 (backbone)
        feats = self.model(imgs)       # e.g. [B, C_feat, Hf, Wf]

        # 3) 동적 kernel 생성 (KernelPredictor)
        kernels = self.kernel_pred(feats)  # [B, 1, k, k]
        pad = kernels.size(-1) // 2        # k//2 (3×3일 때 pad=1)

        # 4) 그룹(depthwise) 컨볼루션을 배치 단위로 병렬 처리
        #    - for-loop 대신, 한 번에 (B×C) 채널로 reshape하여 conv2d 호출
        B, C, H, W = imgs.shape           # C = 3 (RGB)
        k = kernels.size(-1)              # 3

        # (a) kernels: [B, 1, k, k] → [B, C, k, k]
        kernels_expanded = kernels.repeat(1, C, 1, 1)  # [B, C, k, k]

        # (b) conv2d weight 형식을 [B×C, 1, k, k]로 변환 
        #     (각 배치마다, 각 채널에 대해 별도의 필터를 적용하기 위함)
        weight = kernels_expanded.view(B * C, 1, k, k)  # [B*C, 1, k, k]

        # (c) input: [B, C, H, W] → [1, B*C, H, W]
        imgs_reshape = imgs.view(1, B * C, H, W)

        # (d) groups = B*C 로 두면, 채널마다 개별 필터가 적용됨
        blurred = F.conv2d(imgs_reshape, weight, padding=pad, groups=B * C)
        # blurred: [1, B*C, H, W] → [B, C, H, W]
        enhanced = blurred.view(B, C, H, W)

        # 5) Master YOLOv8 raw logits 추출 (NMS 이전)
        #    Ultralytics YOLOv8 내부 모델을 직접 호출하여 raw predictions을 얻는다.
        #    self.yolo.model(enhanced) → list of tensors (각 스케일별 출력)
        #    각 텐서 형태: [B, n_anchors, grid_h, grid_w, 5 + num_classes]
        #    따라서 objectness(logit)는 마지막 dim index 4에 위치한다.
        #Version 수정(Gradient 소실 문제 해결결)
        raw_preds = self.yolo.model(enhanced)

        # YOLOv8 버전에 따라 raw_preds가 Tensor 하나로 올 수도 있고, 
        # 여러 스케일을 리스트로 반환하거나, 배치별로 리스트 형태로 반환할 수도 있음.
        if isinstance(raw_preds, (list, tuple)):
            preds_list = raw_preds
        else:
            preds_list = [raw_preds]

        # 6) 모든 스케일에서 objectness(logit)만 모아서 하나의 긴 텐서로 합침
        conf_logits_list = []

        for p in preds_list:
            # p가 리스트(또는 튜플)인 경우, 그 내부 요소를 한 번 더 반복해서 접근
            if isinstance(p, (list, tuple)):
                for scale_tensor in p:
                    # scale_tensor.shape == [B, n_anchors, grid_h, grid_w, 5 + nc]
                    conf = scale_tensor[..., 4]             # objectness(logit) 부분
                    conf_flat = conf.flatten(1)             # [B, n_anchors*grid_h*grid_w]
                    conf_logits_list.append(conf_flat)
            else:
                # p가 Tensor인 경우
                # p.shape == [B, n_anchors, grid_h, grid_w, 5 + nc]
                conf = p[..., 4]
                conf_flat = conf.flatten(1)
                conf_logits_list.append(conf_flat)

        # [B, total_preds_across_scales]
        all_conf_logits = torch.cat(conf_logits_list, dim=1)  # [B, M]
        # 7) raw confidence 예측값: sigmoid(로짓)
        raw_conf = torch.sigmoid(all_conf_logits)  # [B, M]

        # 8) 이미지당 평균 confidence, 그리고 배치 전체 평균
        #    conf_per_image: [B], mean over M 예측값 
        conf_per_image = raw_conf.mean(dim=1)
        #    batch_mean_conf: scalar 
        batch_mean_conf = conf_per_image.mean()

        # 9) loss 계산: “confidence를 최대화” → “negative confidence”를 최소화
        loss = -batch_mean_conf

        # 10) backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), batch_mean_conf.item()

    def fit(self, dataloader, epochs=10):
        for epoch in range(1, epochs + 1):
            total_loss, total_conf = 0.0, 0.0
            for imgs, _ in dataloader:
                loss, conf = self.train_step(imgs)
                total_loss += loss
                total_conf += conf
            n = len(dataloader)
            print(f"Epoch {epoch}/{epochs} — Loss: {total_loss / n:.4f}, MeanRawConf: {total_conf / n:.4f}")

        print("Training 완료.")

    # (Optional) gradient가 backbone/kernel_pred으로 실제로 흘러드는지 확인하는 예시 함수
    # torch.autograd.gradcheck는 입력을 double precision으로 요구하기 때문에, 
    # 테스트용으로 작동을 확인하기 위해 작은 batch와 작은 네트워크로만 실행해야 합니다.
    def check_gradient_flow(self, single_image):
        """
        single_image: torch.Tensor [1, 3, H, W], dtype=torch.double, requires_grad=True
        - 작은 크기(H,W 비대칭이 아닌, 예: 64×64)로 테스트 권장
        """
        # 임시로 backbone/kernel_pred 파라미터를 double, requires_grad=True로 설정
        self.model.double()
        self.kernel_pred.double()
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.kernel_pred.parameters():
            p.requires_grad = True

        img = single_image  # dtype=torch.double, requires_grad=True
        assert img.dtype == torch.double and img.requires_grad

        # gradcheck를 위해 callback function 정의
        def func(inp):
            # inp: [1,3,H,W] double
            feats = self.model(inp)
            kernels = self.kernel_pred(feats)   # [1,1,3,3]
            pad = kernels.size(-1) // 2
            B, C, H0, W0 = inp.shape

            # depthwise group conv
            kernels_e = kernels.repeat(1, C, 1, 1).view(B * C, 1, 3, 3)
            inp_r = inp.view(1, B * C, H0, W0)
            blurred = F.conv2d(inp_r, kernels_e, padding=pad, groups=B * C)
            enhanced = blurred.view(B, C, H0, W0)

            # YOLO raw logits (단 scale 1개만 있는 상황 가정)
            raw_preds = self.yolo.model(enhanced)  # 강제 train/eval 모드 상황에 맞춰 조정
            preds_list = raw_preds if isinstance(raw_preds, (list, tuple)) else [raw_preds]

            confs_list = []
            for p in preds_list:
                confs_list.append(p[..., 4].flatten(1))  # [B, M]
            all_conf_logits = torch.cat(confs_list, dim=1)  # [B, M]
            raw_conf = torch.sigmoid(all_conf_logits)       # [B, M]
            conf_mean = raw_conf.mean()                     # scalar double
            # loss = -conf_mean
            return -conf_mean

        # gradcheck 수행 (요구사항: 몇 번의 백/포워드 연산이 추가로 발생함)
        test = torch.autograd.gradcheck(func, (img,), eps=1e-3, atol=1e-2)
        print("Gradcheck 결과:", test)

        # 원래 dtype/grad 상태로 되돌리기
        self.model.float()
        self.kernel_pred.float()
        for p in self.model.parameters():
            p.requires_grad = False  # (원래 freeze 상태로 되돌림)
        for p in self.kernel_pred.parameters():
            p.requires_grad = True

        return test
