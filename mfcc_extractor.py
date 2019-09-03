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

#set the path to your audio dataset
import pathlib
data_root = pathlib.Path(r'##PATH TO YOUR RAW AUDIO##\audio')
for item in data_root.iterdir():
  print(item)
  
import random
all_audio_paths = list(data_root.glob('*/*'))
all_audio_paths = [str(path) for path in all_audio_paths]
random.shuffle(all_audio_paths)

file_count = len(all_audio_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_names

# Iterate through each file, convert it to an MFCC image and save it to the apropriate folder. This version of the code requires you to create the empty folders manually first
counter = 0
fig = plt.figure(figsize=(10, 10))
for item in all_audio_paths:
  name = str(item.name).split('.')
  name = name[0] + '.png'
  parent = str(item.parent).split("audio")
  path = r'##PATH TO THE IMGE OUTPUT DESTINATION##\images' + '\\' + parent[1][1:] + '\\' + name
  print(path)
  print(pathlib.Path(path).exists())
  counter = counter + 1
  output = str(counter) + "/" + str(file_count)
  print(output)
  y, sr = librosa.load(item)
  
  # Generate a mel spectrogram for the audio file
  S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

  # Convert to log scale (dB). We'll use the peak power (max) as reference.
  log_S = librosa.power_to_db(S, ref=np.max)

  # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
  mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

  librosa.display.specshow(mfcc)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
  plt.margins(0,0)
  fig.savefig(path, dpi=300, bbox_inches = 'tight',  pad_inches = 0)
  fig.clf()
  gc.collect()
      
