import paddle
import paddle.nn as nn

__all__ = [
    "RMSE",
    "MSE",
]


class RMSE(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = target > 1e-3
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = paddle.sum(err.reshape((err.shape[0], 1, -1)), -1, keepdim=True)
        cnt = paddle.sum(
            val_pixels.reshape((val_pixels.shape[0], 1, -1)), -1, keepdim=True
        )
        return paddle.sqrt(loss / cnt)


class MSE(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = target > 1e-3
        loss = target * val_pixels - outputs * val_pixels
        return loss**2


if __name__ == "__main__":
    # model = MSE()
    model = RMSE()
    outputs = paddle.randn((1, 10), dtype="float16")
    target = paddle.randn((1, 10), dtype="float16")
    out = model(outputs, target)
    print(out)
