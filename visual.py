import matplotlib.pyplot as plt
import pandas as pd


def plot_csv(csv_file, cols, title="", x_label="", y_label="", save=None):
    """
    Plots the contents of a CSV file

    :param csv_file: Name of file of which the contents are plotted
    :param cols: List of strings of the columns that are plotted
    :param title: The title of the plot
    :param x_label: The x-label of the plot
    :param y_label: The y-label of the plot
    :param save: Save name of file figure is saved to (optional)
    """
    df = pd.read_csv(csv_file, header=0, engine="python")
    plt.figure()

    for col in cols:
        plt.plot(df[col], label=col)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    plt.show()


def show_images(imgs, rows, cols, save=None):
    """
    Show/save multiple image in a single figure

    :param imgs: Tensor or list containing images
    :param rows: Number of rows of images
    :param cols: Number of cols of images
    :param save: Name of file figure is saved to (optional)
    """
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.tight_layout(w_pad=1)
    idx = 0
    for i in range(rows * cols):
        axs.ravel()[i].imshow(imgs[idx].permute(1, 2, 0))
        axs.ravel()[i].axes.get_xaxis().set_visible(False)
        axs.ravel()[i].axes.get_yaxis().set_visible(False)
        idx += 1

    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    plt.show()


def show_image(img, save=None):
    """
    Show/save a single image

    :param img: The image that is plotted/saved
    :param save: Name of file figure is saved to (optional)
    """
    fig, axs = plt.subplots(figsize=(3, 3))
    axs.imshow(img.permute(1, 2, 0))
    axs.axes.get_xaxis().set_visible(False)
    axs.axes.get_yaxis().set_visible(False)

    if save is not None:
        plt.savefig(save, bbox_inches='tight')

    plt.show()
