import gradio as gr


def set_preset(config_str):
    choices = [
        "Small Masked Changes (e.g. add lipstick)",
        "Major Masked Changes (e.g. change hair color or nose size)",
        "Major Global Changes (e.g. change race / gender",
    ]
    if config_str == choices[0]:
        return set_small_local()
    elif config_str == choices[1]:
        return set_major_local()
    elif config_str == choices[2]:
        return set_major_global()


def set_small_local():
    return (
        gr.Slider.update(value=25),
        gr.Slider.update(value=0.15),
        gr.Slider.update(value=1),
        gr.Slider.update(value=4),
    )


def set_major_local():
    return (
        gr.Slider.update(value=25),
        gr.Slider.update(value=0.25),
        gr.Slider.update(value=35),
        gr.Slider.update(value=10),
    )


def set_major_global():
    return (
        gr.Slider.update(value=30),
        gr.Slider.update(value=0.1),
        gr.Slider.update(value=2),
        gr.Slider.update(value=0.2),
    )
