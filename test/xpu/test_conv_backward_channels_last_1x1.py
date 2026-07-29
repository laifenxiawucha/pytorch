# Owner(s): ["module: intel"]
#
# Regression test for issue #1966 (candidate pr_190773):
# XPU oneDNN convolution backward silently produced wrong gradients for
# channels-last 1x1 spatial / kernel shapes, because the backward path derived
# its is_channels_last flag from diff_dst (layout-nondiscriminating at 1x1)
# instead of the caller-allocated output buffer (diff_weight / diff_src).
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestConvBackwardChannelsLast1x1(TestCase):
    def test_backward_weights_1x1_spatial_channels_last_weight(self):
        # convolution_backward_weights: 1x1-spatial activation, 3x3 CL weight.
        # This is the discriminating case (weight_grad_diff ~= 8.3 pre-fix).
        torch.manual_seed(42)
        input_ref = torch.randn((8, 16, 1, 1), device="xpu", requires_grad=True)
        weight_ref = torch.randn((16, 16, 3, 3), device="xpu", requires_grad=True)
        grad = torch.randn(
            torch.conv2d(input_ref, weight_ref, padding=1).shape, device="xpu"
        )

        torch.conv2d(input_ref, weight_ref, padding=1).backward(grad)
        expected_weight_grad = weight_ref.grad.clone()

        input_cl = input_ref.detach().requires_grad_()
        weight_cl = (
            weight_ref.detach()
            .to(memory_format=torch.channels_last)
            .requires_grad_()
        )
        torch.conv2d(input_cl, weight_cl, padding=1).backward(grad)

        self.assertEqual(weight_cl.grad.device.type, "xpu")
        self.assertEqual(weight_cl.grad, expected_weight_grad, atol=1e-4, rtol=1e-4)

    def test_backward_data_stride2_1x1_channels_last_input(self):
        # convolution_backward_data: stride-2 1x1 downsample, CL input.
        torch.manual_seed(7)
        input_ref = torch.randn((8, 16, 8, 8), device="xpu", requires_grad=True)
        weight_ref = torch.randn((32, 16, 1, 1), device="xpu", requires_grad=True)
        out = torch.conv2d(input_ref, weight_ref, stride=(2, 2))
        grad = torch.randn(out.shape, device="xpu")

        out.backward(grad)
        expected_input_grad = input_ref.grad.clone()

        input_cl = (
            input_ref.detach()
            .to(memory_format=torch.channels_last)
            .requires_grad_()
        )
        weight_c = weight_ref.detach().requires_grad_()
        torch.conv2d(input_cl, weight_c, stride=(2, 2)).backward(grad)

        self.assertEqual(input_cl.grad.device.type, "xpu")
        self.assertEqual(input_cl.grad, expected_input_grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
