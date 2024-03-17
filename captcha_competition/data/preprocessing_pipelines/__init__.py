from captcha_competition.data.preprocessing_pipelines.steps import (
    remove_background,
    to_grayscale,
    apply_morphological_closing,
    resize,
    image_to_tensor,
    label_to_tensor,
    get_numbers_color,
    get_background_color,
    get_most_common_colors,
)

from captcha_competition.data.preprocessing_pipelines.pipelines import (
    create_preprocessing_pipeline,
    STEPS,
)
