import cv2
import albumentations as A
import numpy as np


def _safe_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


class XRayQuantumNoise(A.ImageOnlyTransform):
    """Low-dose fluoroscopy noise: Poisson quantum noise + mild electronic Gaussian noise."""

    def __init__(
        self,
        dose_range=(6.0, 20.0),
        gaussian_sigma_range=(1.0, 6.0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.dose_range = dose_range
        self.gaussian_sigma_range = gaussian_sigma_range

    def apply(self, img, dose_level=10.0, gaussian_sigma=3.0, **params):
        img_f = img.astype(np.float32) / 255.0
        poisson = np.random.poisson(np.clip(img_f, 0.0, 1.0) * dose_level) / max(dose_level, 1e-6)
        gaussian = np.random.normal(
            loc=0.0,
            scale=gaussian_sigma / 255.0,
            size=img_f.shape,
        ).astype(np.float32)
        noisy = poisson + gaussian
        return _safe_uint8(noisy * 255.0)

    def get_params(self):
        return {
            "dose_level": np.random.uniform(*self.dose_range),
            "gaussian_sigma": np.random.uniform(*self.gaussian_sigma_range),
        }

    def get_transform_init_args_names(self):
        return ("dose_range", "gaussian_sigma_range")


class XRayMotionBlur(A.ImageOnlyTransform):
    """Directional blur to mimic guidewire / C-arm motion during exposure."""

    def __init__(self, degree_range=(5, 19), always_apply=False, p=0.35):
        super().__init__(always_apply=always_apply, p=p)
        self.degree_range = degree_range

    def apply(self, img, degree=9, angle=0.0, **params):
        degree = max(int(degree), 3)
        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[degree // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (degree, degree))
        kernel_sum = kernel.sum()
        if kernel_sum <= 0:
            return img
        kernel /= kernel_sum
        blurred = cv2.filter2D(img, -1, kernel)
        return _safe_uint8(blurred)

    def get_params(self):
        return {
            "degree": int(np.random.randint(self.degree_range[0], self.degree_range[1] + 1)),
            "angle": float(np.random.uniform(0.0, 180.0)),
        }

    def get_transform_init_args_names(self):
        return ("degree_range",)


class XRayScatterField(A.ImageOnlyTransform):
    """Low-frequency intensity drift to mimic scatter, vignetting, and uneven exposure."""

    def __init__(
        self,
        field_strength_range=(0.08, 0.22),
        bias_range=(-18.0, 18.0),
        coarse_size=32,
        always_apply=False,
        p=0.35,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.field_strength_range = field_strength_range
        self.bias_range = bias_range
        self.coarse_size = coarse_size

    def apply(self, img, field_strength=0.1, bias=0.0, seed=0, **params):
        rng = np.random.default_rng(seed)
        coarse = rng.normal(loc=0.0, scale=1.0, size=(self.coarse_size, self.coarse_size)).astype(np.float32)
        coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=1.2)
        field = cv2.resize(coarse, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        field -= field.mean()
        field_std = float(field.std()) if field.std() > 1e-6 else 1.0
        field /= field_std

        img_f = img.astype(np.float32)
        gain = 1.0 + field_strength * field
        shifted = img_f * gain + bias
        return _safe_uint8(shifted)

    def get_params(self):
        return {
            "field_strength": float(np.random.uniform(*self.field_strength_range)),
            "bias": float(np.random.uniform(*self.bias_range)),
            "seed": int(np.random.randint(0, 1_000_000)),
        }

    def get_transform_init_args_names(self):
        return ("field_strength_range", "bias_range", "coarse_size")


def _build_fluoro_image_transforms(
    img_size=None,
    profile="fluoro_hard",
    include_flips=True,
    hflip_p=0.5,
    vflip_p=0.5,
):
    transforms = []
    if img_size is not None:
        transforms.append(A.Resize(img_size[0], img_size[1]))

    if include_flips:
        # 这里把水平/垂直翻转概率拆开，是为了后续做更贴近透视场景的消融。
        # 对导丝任务来说，VerticalFlip 往往比 HorizontalFlip 更激进，
        # 因为它更容易打乱真实的解剖方向与器械运动先验。
        transforms.extend(
            [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
            ]
        )

    if profile == "none":
        return transforms

    if profile == "standard":
        transforms.extend(
            [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
            ]
        )
    elif profile == "fluoro_extreme":
        transforms.extend(
            [
                A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.25, p=0.55),
                A.RandomGamma(gamma_limit=(75, 125), p=0.4),
                XRayScatterField(field_strength_range=(0.10, 0.28), bias_range=(-22.0, 22.0), p=0.45),
                A.OneOf(
                    [
                        XRayMotionBlur(degree_range=(7, 25), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.45,
                ),
                A.OneOf(
                    [
                        XRayQuantumNoise(dose_range=(4.5, 14.0), gaussian_sigma_range=(2.0, 8.0), p=1.0),
                        A.GaussNoise(var_limit=(14.0, 40.0), p=1.0),
                    ],
                    p=0.65,
                ),
            ]
        )
    else:
        transforms.extend(
            [
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.22, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.35),
                XRayScatterField(field_strength_range=(0.08, 0.22), bias_range=(-18.0, 18.0), p=0.35),
                A.OneOf(
                    [
                        XRayMotionBlur(degree_range=(5, 19), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=0.35,
                ),
                A.OneOf(
                    [
                        XRayQuantumNoise(dose_range=(6.0, 20.0), gaussian_sigma_range=(1.0, 6.0), p=1.0),
                        A.GaussNoise(var_limit=(10.0, 28.0), p=1.0),
                    ],
                    p=0.55,
                ),
            ]
        )

    return transforms


def build_fluoro_train_transform(
    seq_len,
    img_size=(512, 512),
    profile="fluoro_hard",
    hflip_p=0.5,
    vflip_p=0.5,
):
    targets = {f"image{i}": "image" for i in range(1, seq_len)}
    targets.update({f"mask{i}": "mask" for i in range(1, seq_len)})
    transforms = _build_fluoro_image_transforms(
        img_size=img_size,
        profile=profile,
        include_flips=True,
        hflip_p=hflip_p,
        vflip_p=vflip_p,
    )
    return A.Compose(transforms, additional_targets=targets)


def build_fluoro_image_only_transform(
    num_frames,
    img_size=None,
    profile="fluoro_extreme",
    include_flips=False,
    hflip_p=0.5,
    vflip_p=0.5,
):
    targets = {f"image{i}": "image" for i in range(1, num_frames)}
    transforms = _build_fluoro_image_transforms(
        img_size=img_size,
        profile=profile,
        include_flips=include_flips,
        hflip_p=hflip_p,
        vflip_p=vflip_p,
    )
    return A.Compose(transforms, additional_targets=targets)


def build_fluoro_replay_transform(
    img_size=None,
    profile="fluoro_extreme",
    include_flips=False,
    hflip_p=0.5,
    vflip_p=0.5,
):
    # ReplayCompose 适合离线生成 aug_data：先在一帧上采样参数，再稳定回放到整段序列。
    transforms = _build_fluoro_image_transforms(
        img_size=img_size,
        profile=profile,
        include_flips=include_flips,
        hflip_p=hflip_p,
        vflip_p=vflip_p,
    )
    return A.ReplayCompose(transforms)
