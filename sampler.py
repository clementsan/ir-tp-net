import torch
from torchio.data.subject import Subject
from torchio.data.sampler import RandomSampler
from typing import Generator
import numpy as np


class MyUniformSampler(RandomSampler):
	"""Randomly extract patches from a volume with uniform probability.
	Args:
		patch_size: See :class:`~torchio.data.PatchSampler`.
	"""

	def __init__(self, patch_size, tile_size):
		super().__init__(patch_size)
		self.tile_size = tile_size

	def get_probability_map(self, subject: Subject) -> torch.Tensor:
		return torch.ones(1, *subject.spatial_shape)

	def _generate_patches(
			self,
			subject: Subject,
			num_patches: int = None,
			) -> Generator[Subject, None, None]:
		valid_range = subject.spatial_shape - self.patch_size

		patches_left = num_patches if num_patches is not None else True
		# Random location using tile_size (multiple of tile_size)
		while patches_left:
			index_ini = [
				self.tile_size * torch.randint(x//self.tile_size + 1, (1,)).item()
				for x in valid_range
			]
			index_ini_array = np.asarray(index_ini)
			yield self.extract_patch(subject, index_ini_array)
			if num_patches is not None:
				patches_left -= 1
