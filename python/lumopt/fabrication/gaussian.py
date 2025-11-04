import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

from .base import FabricationModel


class GaussianModel(FabricationModel):
    def __init__(
        self,
        blur_radius_nm: float,
        grid_spacing_nm: float,
        beta: float = 100.0,
        eta: float = 0.5,
        gradient_method: str = "analytical",
    ):
        self.blur_radius_nm = float(blur_radius_nm)
        self.padding_nm = self.blur_radius_nm
        self.grid_spacing_nm = float(grid_spacing_nm)
        self.sigma = self.blur_radius_nm / self.grid_spacing_nm
        self.beta = float(beta)
        self.eta = float(eta)
        self.gradient_method = gradient_method

        self._cached_points = None
        self._cached_raster = None
        self._cached_blurred = None
        self._cached_binary = None
        self._cached_edge_indices = None
        self._cached_coord_info = None

    def apply(self, points: np.ndarray, visualize: bool = False) -> np.ndarray:
        self._cached_points = points.copy()

        self._cached_raster = self._rasterize(points)
        self._cached_blurred = self._gaussian_blur(self._cached_raster)
        self._cached_binary = self._binarize(self._cached_blurred)

        deraster = self._derasterize(self._cached_binary, points)

        if visualize:
            self.plot_fabrication_effects()

        return deraster

    def gradient(self, points: np.ndarray, gradient: list) -> np.ndarray:
        if self.gradient_method == "analytical":
            return self._gradient_analytical(points, gradient)
        elif self.gradient_method == "finite_difference":
            return self._gradient_finite_difference(points, gradient)
        elif self.gradient_method == "none":
            return gradient
        else:
            raise ValueError(f"Unknown gradient method: {self.gradient_method}")

    def _gradient_analytical(self, points: np.ndarray, gradient: list) -> np.ndarray:
        if self._cached_points is not None and np.array_equal(
            points, self._cached_points
        ):
            raster = self._cached_raster
            blurred = self._cached_blurred
            binary = self._cached_binary
        else:
            raster = self._rasterize(points)
            blurred = self._gaussian_blur(raster)
            binary = self._binarize(blurred)
            _ = self._derasterize(binary, points)

        n_points = len(points)
        n_wavelengths = len(gradient[0])
        gradient_array = np.stack(gradient, axis=0)
        reshaped_gradient = np.zeros((n_points, 2, n_wavelengths))
        reshaped_gradient[:, 0, :] = gradient_array[::2]
        reshaped_gradient[:, 1, :] = gradient_array[1::2]

        # Some temporary timing
        t1 = time.time()
        dbinary = self._derasterize_gradient(binary, reshaped_gradient)
        t2 = time.time()
        dblurred = self._binarize_gradient(blurred, dbinary)
        t3 = time.time()
        draster = self._gaussian_blur_gradient(dblurred)
        t4 = time.time()
        points_gradient = self._rasterize_gradient(points, draster)
        t5 = time.time()
        print("Time taken:")
        print(f"    derasterize_gradient: {t2 - t1}s")
        print(f"    binarize_gradient: {t3 - t2}s")
        print(f"    gaussian_blur_gradient: {t4 - t3}s")
        print(f"    rasterize_gradient: {t5 - t4}s")

        flattened_gradient = np.zeros((2 * n_points, n_wavelengths))
        flattened_gradient[::2] = points_gradient[:, 0]
        flattened_gradient[1::2] = points_gradient[:, 1]

        return flattened_gradient

    def _gradient_finite_difference(
        self, points: np.ndarray, gradient: list
    ) -> np.ndarray:
        n_points = len(points)
        n_wavelengths = len(gradient[0])
        gradient_array = np.stack(gradient, axis=0)

        h = self.grid_spacing_nm * 1e-9

        fd_gradient = np.zeros((2 * n_points, n_wavelengths))

        for i in range(n_points):
            for coord in range(2):
                points_plus = points.copy()
                points_plus[i, coord] += h

                fab_nominal = self.apply(points, visualize=False)
                fab_plus = self.apply(points_plus, visualize=False)

                diff = (fab_plus - fab_nominal) / h

                reshaped_gradient = np.zeros((n_points, 2, n_wavelengths))
                reshaped_gradient[:, 0, :] = gradient_array[::2]
                reshaped_gradient[:, 1, :] = gradient_array[1::2]

                for w in range(n_wavelengths):
                    fd_gradient[2 * i + coord, w] = np.sum(
                        diff * reshaped_gradient[:, :, w]
                    )

        return fd_gradient

    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image, sigma=self.sigma)

    def _gaussian_blur_gradient(self, gradient_image: np.ndarray) -> np.ndarray:
        n_wavelengths = gradient_image.shape[2]
        blurred_gradient = np.zeros_like(gradient_image)
        for w in range(n_wavelengths):
            blurred_gradient[:, :, w] = gaussian_filter(
                gradient_image[:, :, w], sigma=self.sigma
            )
        return blurred_gradient

    def _rasterize(self, points: np.ndarray) -> np.ndarray:
        points_nm = points * 1e9

        min_x, min_y = np.min(points_nm, axis=0)
        max_x, max_y = np.max(points_nm, axis=0)

        min_x -= self.padding_nm
        min_y -= self.padding_nm
        max_x += self.padding_nm
        max_y += self.padding_nm

        x = np.arange(min_x, max_x + self.grid_spacing_nm, self.grid_spacing_nm)
        y = np.arange(min_y, max_y + self.grid_spacing_nm, self.grid_spacing_nm)
        X, Y = np.meshgrid(x, y)

        polygon_path = Path(points)

        grid_points = np.column_stack((X.ravel() * 1e-9, Y.ravel() * 1e-9))
        mask = polygon_path.contains_points(grid_points)
        mask = mask.reshape(X.shape)

        return mask.astype(float)

    def _rasterize_gradient(
        self, points: np.ndarray, upstream_gradient: np.ndarray
    ) -> np.ndarray:
        points_nm = points * 1e9
        n_wavelengths = upstream_gradient.shape[2]
        n_points = len(points)

        min_x, min_y = np.min(points_nm, axis=0)
        max_x, max_y = np.max(points_nm, axis=0)
        min_x -= self.padding_nm
        min_y -= self.padding_nm
        max_x += self.padding_nm
        max_y += self.padding_nm
        x = np.arange(min_x, max_x + self.grid_spacing_nm, self.grid_spacing_nm)
        y = np.arange(min_y, max_y + self.grid_spacing_nm, self.grid_spacing_nm)
        X, Y = np.meshgrid(x, y)

        grid_points_m = np.stack([X, Y], axis=-1) * 1e-9
        points_gradient_all = np.zeros((n_points, 2, n_wavelengths))
        epsilon = self.grid_spacing_nm * 1e-9

        edges = points[(np.arange(n_points) + 1) % n_points] - points
        edge_lengths = np.linalg.norm(edges, axis=1)
        edge_units = edges / (edge_lengths[:, None] + 1e-10)

        influence_radius = max(5, int(3 * self.sigma))  # 3 sigma or minimum 5 cells
        influence_radius_m = influence_radius * self.grid_spacing_nm * 1e-9

        for w in range(n_wavelengths):
            points_gradient = np.zeros_like(points)

            for i in range(n_points):
                point_x = (points[i, 0] - min_x * 1e-9) / (self.grid_spacing_nm * 1e-9)
                point_y = (points[i, 1] - min_y * 1e-9) / (self.grid_spacing_nm * 1e-9)

                x_min = max(0, int(point_x - influence_radius))
                x_max = min(X.shape[1], int(point_x + influence_radius + 1))
                y_min = max(0, int(point_y - influence_radius))
                y_max = min(X.shape[0], int(point_y + influence_radius + 1))

                local_grid = grid_points_m[y_min:y_max, x_min:x_max]
                local_grid_flat = local_grid.reshape(-1, 2)
                local_upstream = upstream_gradient[y_min:y_max, x_min:x_max, w]

                grid_to_p1 = local_grid_flat - points[i]
                distances = np.linalg.norm(grid_to_p1, axis=1)

                mask = distances < influence_radius_m
                if not np.any(mask):
                    continue

                grid_to_p1 = grid_to_p1[mask]
                distances = distances[mask]

                proj_length = np.sum(grid_to_p1 * edge_units[i], axis=1)
                proj_length = np.clip(proj_length, 0, edge_lengths[i])

                closest_points = points[i] + proj_length[:, None] * edge_units[i]
                dist_vectors = local_grid_flat[mask] - closest_points

                delta = np.exp(-((distances / epsilon) ** 2))
                weighted_delta = delta * local_upstream.flatten()[mask]

                dist_norms = distances[:, None]
                dist_vectors_normalized = np.divide(
                    dist_vectors,
                    dist_norms + 1e-10,
                    out=np.zeros_like(dist_vectors),
                    where=dist_norms > 1e-10,
                )

                t = proj_length / (edge_lengths[i] + 1e-10)

                weighted_contributions = (
                    weighted_delta[:, None] * dist_vectors_normalized
                )
                points_gradient[i] += np.sum(
                    (1 - t)[:, None] * weighted_contributions, axis=0
                )
                points_gradient[(i + 1) % n_points] += np.sum(
                    t[:, None] * weighted_contributions, axis=0
                )

            points_gradient_all[:, :, w] = points_gradient

        return points_gradient_all

    def _derasterize(
        self, binary: np.ndarray, original_points: np.ndarray
    ) -> np.ndarray:
        # Sobel filter is 3x3 (fixed); casting to float64 ensures the output is 64-bit float
        sobelx = ndimage.sobel(binary.astype(np.float64), axis=1)
        sobely = ndimage.sobel(binary.astype(np.float64), axis=0)
        edges = np.sqrt(sobelx**2 + sobely**2)

        edges = (edges > 0.05 * np.max(edges)).astype(np.uint8)

        min_x, min_y = np.min(original_points * 1e9, axis=0)
        max_x, max_y = np.max(original_points * 1e9, axis=0)

        x_scale = (max_x - min_x + 2 * self.padding_nm) / binary.shape[1]
        y_scale = (max_y - min_y + 2 * self.padding_nm) / binary.shape[0]

        self._cached_coord_info = (min_x, min_y, x_scale, y_scale)

        def physical_to_image(points_nm):
            x_img = (points_nm[:, 0] - (min_x - self.padding_nm)) / x_scale
            y_img = (points_nm[:, 1] - (min_y - self.padding_nm)) / y_scale
            return np.column_stack([x_img, y_img])

        def image_to_physical(points_img):
            x_phys = points_img[:, 0] * x_scale + (min_x - self.padding_nm)
            y_phys = points_img[:, 1] * y_scale + (min_y - self.padding_nm)
            return np.column_stack([x_phys, y_phys])

        points_img = physical_to_image(original_points * 1e9)

        new_points_img = np.zeros_like(points_img)
        edge_indices = np.zeros_like(points_img, dtype=int)

        search_radius = int(max(10, self.sigma * 3))

        y_grid, x_grid = np.ogrid[
            -search_radius : search_radius + 1, -search_radius : search_radius + 1
        ]
        distances = np.sqrt(x_grid**2 + y_grid**2)
        radial_mask = distances <= search_radius

        for i, point in enumerate(points_img):
            x, y = int(round(point[0])), int(round(point[1]))
            x = np.clip(x, 0, binary.shape[1] - 1)
            y = np.clip(y, 0, binary.shape[0] - 1)

            x_min = max(0, x - search_radius)
            x_max = min(binary.shape[1], x + search_radius + 1)
            y_min = max(0, y - search_radius)
            y_max = min(binary.shape[0], y + search_radius + 1)

            local_edges = edges[y_min:y_max, x_min:x_max]

            edge_points = np.where(
                local_edges
                & radial_mask[: local_edges.shape[0], : local_edges.shape[1]]
            )

            if len(edge_points[0]) > 0:
                edge_y_coords = edge_points[0] + y_min
                edge_x_coords = edge_points[1] + x_min

                distances = np.sqrt(
                    (edge_y_coords - point[1]) ** 2 + (edge_x_coords - point[0]) ** 2
                )

                nearest_idx = np.argmin(distances)
                nearest_y = edge_y_coords[nearest_idx]
                nearest_x = edge_x_coords[nearest_idx]

                new_points_img[i] = [nearest_x, nearest_y]
                edge_indices[i] = [nearest_y, nearest_x]
            else:
                extended_radius = int(search_radius * 2)

                x_min = max(0, x - extended_radius)
                x_max = min(binary.shape[1], x + extended_radius + 1)
                y_min = max(0, y - extended_radius)
                y_max = min(binary.shape[0], y + extended_radius + 1)

                local_edges_ext = edges[y_min:y_max, x_min:x_max]
                edge_points = np.where(local_edges_ext)

                if len(edge_points[0]) > 0:
                    edge_y_coords = edge_points[0] + y_min
                    edge_x_coords = edge_points[1] + x_min

                    distances = np.sqrt(
                        (edge_y_coords - point[1]) ** 2
                        + (edge_x_coords - point[0]) ** 2
                    )
                    nearest_idx = np.argmin(distances)
                    nearest_y = edge_y_coords[nearest_idx]
                    nearest_x = edge_x_coords[nearest_idx]

                    new_points_img[i] = [nearest_x, nearest_y]
                    edge_indices[i] = [nearest_y, nearest_x]
                else:
                    new_points_img[i] = point
                    edge_indices[i] = [int(round(point[1])), x]

        self._cached_edge_indices = edge_indices

        new_points_nm = image_to_physical(new_points_img)
        new_points = new_points_nm * 1e-9

        return new_points

    def _derasterize_gradient(
        self, binary: np.ndarray, upstream_gradient: np.ndarray
    ) -> np.ndarray:
        n_wavelengths = upstream_gradient.shape[2]
        n_points = upstream_gradient.shape[0]
        binary_gradient = np.zeros((binary.shape[0], binary.shape[1], n_wavelengths))

        if self._cached_edge_indices is None or self._cached_coord_info is None:
            raise RuntimeError(
                "Cached edge indices or coordinate info not available. "
                "Ensure _derasterize was called before _derasterize_gradient."
            )
        edge_indices = self._cached_edge_indices
        _, _, x_scale, y_scale = self._cached_coord_info

        # Sobel filter is 3x3 (fixed); casting to float64 ensures the output is 64-bit float
        sobelx = ndimage.sobel(binary.astype(np.float64), axis=1)
        sobely = ndimage.sobel(binary.astype(np.float64), axis=0)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        norm_y = np.divide(
            sobely, magnitude, out=np.zeros_like(sobely), where=magnitude > 1e-10
        )

        for i in range(n_points):
            row, col = edge_indices[i]

            row = np.clip(row, 0, binary.shape[0] - 1)
            col = np.clip(col, 0, binary.shape[1] - 1)

            ny = norm_y[row, col]

            for w in range(n_wavelengths):
                g_phys_y = upstream_gradient[i, 1, w]
                g_img_y = g_phys_y * y_scale

                g_normal = g_img_y * ny

                binary_gradient[row, col, w] += g_normal

        return binary_gradient

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        numerator = np.tanh(self.beta * self.eta) + np.tanh(self.beta * (x - self.eta))
        denominator = np.tanh(self.beta * self.eta) + np.tanh(
            self.beta * (1 - self.eta)
        )
        return numerator / denominator

    def _binarize_gradient(
        self, x: np.ndarray, upstream_gradient: np.ndarray
    ) -> np.ndarray:
        n_wavelengths = upstream_gradient.shape[2]
        gradient = (
            self.beta
            * (1 - np.tanh(self.beta * (x - self.eta)) ** 2)
            / (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (1 - self.eta)))
        )

        gradient_all = np.zeros_like(upstream_gradient)
        for w in range(n_wavelengths):
            gradient_all[:, :, w] = gradient * upstream_gradient[:, :, w]

        return gradient_all

    def plot_fabrication_effects(self) -> None:
        if self._cached_points is None:
            raise RuntimeError("No cached data available. Run apply() first.")

        _, axes = plt.subplots(2, 2, figsize=(8, 7))

        points_nm = self._cached_points * 1e9
        deraster = self._derasterize(self._cached_binary, self._cached_points)
        deraster_nm = deraster * 1e9

        min_x, min_y = np.min(points_nm, axis=0)
        max_x, max_y = np.max(points_nm, axis=0)
        extent = [
            min_x - self.padding_nm,
            max_x + self.padding_nm,
            min_y - self.padding_nm,
            max_y + self.padding_nm,
        ]

        ax = axes[0, 0]
        ax.imshow(self._cached_raster, origin="lower", cmap="gray", extent=extent)
        ax.plot(points_nm[:, 0], points_nm[:, 1], "g-", label="Original", linewidth=2)
        ax.set_title("Rasterization")

        ax = axes[0, 1]
        ax.imshow(self._cached_blurred, origin="lower", cmap="gray", extent=extent)
        ax.set_title("Fabrication")

        ax = axes[1, 0]
        ax.imshow(self._cached_binary, origin="lower", cmap="gray", extent=extent)
        ax.set_title("Binarization")

        ax = axes[1, 1]
        ax.imshow(
            self._cached_binary, origin="lower", cmap="gray", alpha=0.5, extent=extent
        )
        ax.plot(
            deraster_nm[:, 0],
            deraster_nm[:, 1],
            "r--",
            label="Derasterized",
            linewidth=2,
        )
        ax.plot(
            points_nm[:, 0],
            points_nm[:, 1],
            "g-",
            alpha=0.5,
            label="Original",
            linewidth=2,
        )
        ax.set_title("Derasterization")

        for ax in axes.flat:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("auto")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
