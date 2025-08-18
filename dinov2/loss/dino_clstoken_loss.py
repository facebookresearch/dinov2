import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class DINOLoss(nn.Module):
    """Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature parameter for the teacher network.
        student_temp:
            Temperature parameter for the student network.
        center:
            Center used for the teacher output. It is updated with a moving average
            during training.
        center_momentum:
            Momentum term for the center calculation.
        warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        teacher_temp_schedule:
            A linear schedule for the teacher temperature during the warmup phase (for backward compatibility).

    Examples:
        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate a view of the images with a random transform
        >>> view = transform(images)
        >>>
        >>> # embed the view with a student and teacher model
        >>> teacher_out = teacher(view)
        >>> student_out = student(view)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn([teacher_out], [student_out])
    """

    def __init__(
        self,
        out_dim: int = 65536,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the DINOLoss Module.

        Args:
            center_mode:
                Mode for center calculation. Only 'mean' is supported.
            warmup_teacher_temp:
                Initial temperature for the teacher network (for backward compatibility).
            warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_output_list: list[Tensor],
        teacher_out_softmaxed_centered_list: list[Tensor],
        graph=None,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks.

        Returns:
            The average cross-entropy loss.
        """

        def normalize_to_list(x):
            if torch.is_tensor(x):
                return [x]
            if isinstance(x, (list, tuple)):
                return [t for t in x if torch.is_tensor(t)]
            raise TypeError(f"Unsupported type for stacking: {type(x)}")

        # Calculate cross-entropy loss.
        if len(teacher_out_softmaxed_centered_list) == len(student_output_list):
            teacher_out_softmaxed_centered_list = list(teacher_out_softmaxed_centered_list[0].chunk(2, dim=0)) [::-1]
            student_output_list = list(student_output_list[0].chunk(2, dim=0))
        teacher_out_softmaxed_centered_list = normalize_to_list(
            teacher_out_softmaxed_centered_list
        )
        student_output_list = normalize_to_list(student_output_list)
        t_out = torch.stack(teacher_out_softmaxed_centered_list)
        student_out_stacked = torch.stack(student_output_list)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        t_out = t_out.squeeze()
        s_out = s_out.squeeze()
        if graph is not None:
            bs = t_out.shape[1]
            graph = graph.view(t_out.shape[0], bs, s_out.shape[0], bs)
            graph = graph.float()
            graph = graph.to(t_out.device)
            # print(f"Graph shape: {graph.shape}")
            loss = - torch.einsum('tbd,tbsb,sbd->ts', t_out, graph,s_out.float(),)
        else:
            loss = -torch.einsum("tbd,sbd->ts", t_out, s_out.float())
        loss.fill_diagonal_(0)

        # Number of loss terms, ignoring the diagonal
        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = t_out.shape[1]

        loss = loss.sum() / (n_terms * batch_size)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True
