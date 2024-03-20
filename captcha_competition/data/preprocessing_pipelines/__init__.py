from captcha_competition.data.preprocessing_pipelines.steps import (
    remove_background,
    to_grayscale,
    min_max_normalize,
    apply_morphological_closing,
    resize,
    image_to_tensor,
    label_to_tensor,
    get_numbers_color,
    get_background_color,
    get_most_common_colors,
    remove_background_tensor,
    to_grayscale_tensor,
    min_max_normalize_tensor,
    resize_tensor,
    get_numbers_color_tensor,
    get_background_color_tensor,
    get_most_common_colors_tensor,
    remove_only_background_tensor,
    preprocessing_tensor,
)

from captcha_competition.data.preprocessing_pipelines.pipelines import (
    create_preprocessing_pipeline,
    STEPS,
)
