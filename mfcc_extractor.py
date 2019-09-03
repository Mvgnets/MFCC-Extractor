# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
%matplotlib inline

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

#import garbage collector to prevent memory leak
import gc

from google.colab import drive
drive.mount('/content/drive')

import pathlib
#data_root = pathlib.Path('/content/drive/My Drive/Audio')
#data_root = pathlib.Path(r'\Users\EchoY\OneDrive\Desktop\Project Data\train\audio')
#for item in data_root.iterdir():
#  print(item)
image_root = pathlib.Path(r'\Users\EchoY\OneDrive\Desktop\Project Data\train\images2')
for item in image_root.iterdir():
  print(item)
  
import random
#all_audio_paths = list(data_root.glob('*/*'))
all_image_paths = list(image_root.glob('*/*'))
#all_audio_paths = [str(path) for path in all_audio_paths]
random.shuffle(all_audio_paths)

#file_count = len(all_audio_paths)
file_count = len(all_image_paths)
file_count

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_names

counter = 0
fig = plt.figure(figsize=(10, 10))
for item in all_audio_paths:
  name = str(item.name).split('.')
  name = name[0] + '.png'
  parent = str(item.parent).split("audio")
  path = r'\Users\EchoY\OneDrive\Desktop\Project Data\train\images' + '\\' + parent[1][1:] + '\\' + name
  print(path)
  print(pathlib.Path(path).exists())
  counter = counter + 1
  output = str(counter) + "/" + str(file_count)
  print(output)
  y, sr = librosa.load(item)
  S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

  # Convert to log scale (dB). We'll use the peak power (max) as reference.
  log_S = librosa.power_to_db(S, ref=np.max)

  # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
  mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

  # How do they look?  We'll show each in its own subplot

  librosa.display.specshow(mfcc)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
  plt.margins(0,0)
  fig.savefig(path, dpi=300, bbox_inches = 'tight',  pad_inches = 0)
  fig.clf()
  gc.collect()
      
