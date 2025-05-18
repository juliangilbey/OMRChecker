import os

import cv2
import numpy as np

from src.logger import logger
from src.processors.interfaces.ImagePreprocessor import ImagePreprocessor
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils


CORNERS = ("UL", "UR", "LL", "LR")


class CropOnDistinctMarkers(ImagePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.tuning_config
        marker_ops = self.options
        self.threshold_circles = []
        # img_utils = ImageUtils()

        # options with defaults
        self.marker_paths = {}
        for corner in CORNERS:
            self.marker_paths[corner] = os.path.join(
                self.relative_dir,
                marker_ops.get(f"relativePath{corner}", f"omr_marker_{corner}.jpg"),
            )
        self.min_matching_threshold = marker_ops.get("min_matching_threshold", 0.3)
        self.max_matching_variation = marker_ops.get("max_matching_variation", 0.41)
        self.marker_rescale_range = tuple(
            int(r) for r in marker_ops.get("marker_rescale_range", (35, 100))
        )
        self.marker_rescale_steps = int(marker_ops.get("marker_rescale_steps", 10))
        self.apply_erode_subtract = marker_ops.get("apply_erode_subtract", True)
        self.markers = self.load_markers(marker_ops, config)

    def __str__(self):
        return f"<{self.marker_paths['UL']}, {self.marker_paths['UR']}, {self.marker_paths['LL']}, {self.marker_paths['LR']}>"

    def exclude_files(self):
        return list(self.marker_paths.values())

    def apply_filter(self, image, file_path):
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        image_eroded_sub = ImageUtils.normalize_util(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
         # Quads on warped image
        quads = {}
        origins = {}
        h1, w1 = image_eroded_sub.shape[:2]
        midh, midw = h1 // 3, w1 // 2
        quads["UL"] = image_eroded_sub[0:midh, 0:midw]
        origins["UL"] = [0, 0]
        quads["UR"] = image_eroded_sub[0:midh, midw:w1]
        origins["UR"] = [midw, 0]
        quads["LL"] = image_eroded_sub[midh:h1, 0:midw]
        origins["LL"] = [0, midh]
        quads["LR"] = image_eroded_sub[midh:h1, midw:w1]
        origins["LR"] = [midw, midh]

        # Draw Quadlines
        image_eroded_sub[:, midw : midw + 2] = 255
        image_eroded_sub[midh : midh + 2, :] = 255

        best_scale, all_max_t = self.getBestMatch(quads)
        if best_scale is None:
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("Quads", image_eroded_sub, config=config)
            return None

        optimal_markers = {}
        _h = {}
        w = {}
        for corner in CORNERS:
            optimal_markers[corner] = ImageUtils.resize_util_h(
                self.markers[corner],
                u_height=int(self.markers[corner].shape[0] * best_scale),
            )
            _h[corner], w[corner] = optimal_markers[corner].shape[:2]
        outer_corners = []
        # add on (width, height) multiple of _h and w compared to top left
        # corner of marker
        outer_corner_factors = {"UL": [0, 0], "UR": [1, 0], "LL": [0, 1], "LR": [1, 1]}
        sum_t, max_t = 0, 0
        quarter_match_log = "Matching Marker:  "
        for corner in CORNERS:
            res = cv2.matchTemplate(
                quads[corner], optimal_markers[corner], cv2.TM_CCOEFF_NORMED
            )
            max_t = res.max()
            quarter_match_log += f"Quarter {corner}: {str(round(max_t, 3))}\t"
            if (
                max_t < self.min_matching_threshold
                or abs(all_max_t - max_t) >= self.max_matching_variation
            ):
                logger.error(
                    file_path,
                    "\nError: No marker found in Quad",
                    corner,
                    "\n\t min_matching_threshold",
                    self.min_matching_threshold,
                    "\t max_matching_variation",
                    self.max_matching_variation,
                    "\t max_t",
                    max_t,
                    "\t all_max_t",
                    all_max_t,
                )
                if config.outputs.show_image_level >= 1:
                    InteractionUtils.show(
                        f"No markers: {file_path}",
                        image_eroded_sub,
                        0,
                        config=config,
                    )
                    InteractionUtils.show(
                        f"res_{corner} ({str(max_t)})",
                        res,
                        1,
                        config=config,
                    )
                return None

            pt = np.argwhere(res == max_t)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[corner][0]
            pt[1] += origins[corner][1]
            # print(">>",pt)
            image = cv2.rectangle(
                image,
                tuple(pt),
                (pt[0] + w[corner], pt[1] + _h[corner]),
                (150, 150, 150),
                2,
            )
            # display:
            image_eroded_sub = cv2.rectangle(
                image_eroded_sub,
                tuple(pt),
                (pt[0] + w[corner], pt[1] + _h[corner]),
                (50, 50, 50) if self.apply_erode_subtract else (155, 155, 155),
                4,
            )
            outer_corners.append(
                [
                    pt[0] + w[corner] * outer_corner_factors[corner][0],
                    pt[1] + _h[corner] * outer_corner_factors[corner][1],
                ]
            )
            sum_t += max_t

        logger.info(quarter_match_log)
        logger.info(f"Optimal Scale: {best_scale}")
        # analysis data
        self.threshold_circles.append(sum_t / 4)

        image = ImageUtils.four_point_transform(image, np.array(outer_corners))
        # appendSaveImg(1,image_eroded_sub)
        # appendSaveImg(1,image_norm)

        image_instance_ops.append_save_img(2, image_eroded_sub)
        # Debugging image -
        # res = cv2.matchTemplate(image_eroded_sub,optimal_marker,cv2.TM_CCOEFF_NORMED)
        # res[ : , midw:midw+2] = 255
        # res[ midh:midh+2, : ] = 255
        # show("Markers Matching",res)
        if config.outputs.show_image_level >= 2 and config.outputs.show_image_level < 4:
            image_eroded_sub = ImageUtils.resize_util_h(
                image_eroded_sub, image.shape[0]
            )
            image_eroded_sub[:, -5:] = 0
            h_stack = np.hstack((image_eroded_sub, image))
            InteractionUtils.show(
                f"Warped: {file_path}",
                ImageUtils.resize_util(
                    h_stack, int(config.dimensions.display_width * 1.6)
                ),
                0,
                0,
                [0, 0],
                config=config,
            )
        # iterations : Tuned to 2.
        # image_eroded_sub = image_norm - cv2.erode(image_norm, kernel=np.ones((5,5)),iterations=2)
        return image

    def load_markers(self, marker_ops, config):
        markers = {}

        for corner in CORNERS:
            if not os.path.exists(self.marker_paths[corner]):
                logger.error(
                    "Marker not found at path provided in template:",
                    self.marker_paths[corner],
                )
                exit(31)

            marker = cv2.imread(self.marker_paths[corner], cv2.IMREAD_GRAYSCALE)

            if "sheetToMarkerWidthRatio" in marker_ops:
                marker = ImageUtils.resize_util(
                    marker,
                    config.dimensions.processing_width
                    / int(marker_ops["sheetToMarkerWidthRatio"]),
                )
            marker = cv2.GaussianBlur(marker, (5, 5), 0)
            marker = cv2.normalize(
                marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
            )

            if self.apply_erode_subtract:
                marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)
            markers[corner] = marker

        return markers

    # Resizing the marker within scaleRange at rate of descent_per_step to
    # find the best match.
    def getBestMatch(self, quads):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        _h = {}
        for corner in CORNERS:
            _h[corner] = self.markers[corner].shape[0]
        res, best_scale = None, None
        all_max_t = 0

        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):  # reverse order
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
            max_ts = []
            for corner in CORNERS:
                rescaled_marker = ImageUtils.resize_util_h(
                    self.markers[corner], u_height=int(_h[corner] * s)
                )
                # res is the black image with white dots
                res = cv2.matchTemplate(
                    quads[corner], rescaled_marker, cv2.TM_CCOEFF_NORMED
                )

                max_ts.append(res.max())

            max_t = max(max_ts)
            if all_max_t < max_t:
                # print('Scale: '+str(s)+', Circle Match: '+str(round(max_t*100,2))+'%')
                best_scale, all_max_t = s, max_t

        if all_max_t < self.min_matching_threshold:
            logger.warning(
                "\tTemplate matching too low! Consider rechecking preProcessors applied before this."
            )
            if config.outputs.show_image_level >= 1:
                InteractionUtils.show("res", res, 1, 0, config=config)

        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange:", self.marker_rescale_range
            )
        return best_scale, all_max_t
