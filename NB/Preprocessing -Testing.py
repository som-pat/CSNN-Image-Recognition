import numpy as np
import cv2
import pyNN.spiNNaker as sim

def preprocess(image, size=(32, 32)):
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to be between 0 and 1
    image = image.astype(np.float32) / 255.0

    # Convert the image to a spike train
    spike_train = sim.SpikeSourcePoisson(
        rates=image.flatten() * 1000, duration=sim_time_step)

    # Apply spatial filtering to the spike train
    kernel = np.ones((3, 3), np.float32) / 9
    filtered_spikes = cv2.filter2D(
        spike_train.getSpikes(), -1, kernel).reshape(size)

    # Apply temporal filtering to the spike train
    filtered_spikes = sim.StaticSynapse().filter(filtered_spikes)

    return filtered_spikes
