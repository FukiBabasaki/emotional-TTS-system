from sound_creator import SoundCreator
from fujisaki_extractor import FujisakiExtractor
from feature_extractor import FeatureExtractor
from resynthesiser import Resynthesiser
import os, shutil
import pandas as pd

class TTSCreator:
    """
    A class representing TTS creator. It maintains all processes to create the resynthesised speech.
    """
    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion
        self.sound_creator = SoundCreator(text)
        self.fujisaki_extractor = FujisakiExtractor(text)
        self.feature_extractor = FeatureExtractor(text)
        self.resynthesiser = Resynthesiser(emotion)


    def create_speech(self):
        """
        This function run several functions to make the resynthesis work.
        1. Get sound from MaryTTS (neutral)
        2. Extract fujisaki parameters from the sound file
        3. Extract text features from txt file.
        4. Merge fujisaki parameters and text features into a csv for prediction
        5. predict new fujisaki parameters and create new wave file.

        The new wave file will be located in app/sound folder.
        """

        self.sound_file = self.sound_creator.create_sound()
        fujisaki_features = self.fujisaki_extractor.extract_features(self.sound_file)
        text_features = self.feature_extractor.extract_text_features()
        merged_csv = self.merge_textfeat_and_fujifeat(fujisaki_features, text_features)
        pitchtier = self.resynthesiser.resynthesis(merged_csv, self.fujisaki_extractor.pac, self.fujisaki_extractor.lab)

        self.generate_wav_from_pitchtier(self.sound_file, pitchtier)
        
        self.clean_up()

        return "input: " + self.text + " " + self.emotion

    def merge_textfeat_and_fujifeat(self, fuji, textfeat):
        """
        This function takes path to fujisaki parameters csv file and text features csv file and merges them into
        a csv file called text_and_fujifeat.csv.
        """
        dfs = [pd.read_csv(fuji), pd.read_csv(textfeat)]
        df_merged = pd.concat(dfs, axis=1)
        df_merged.to_csv('text_and_fujifeat.csv')

        return os.path.relpath('text_and_fujifeat.csv')

    def generate_wav_from_pitchtier(self, wave, pitchtier):
        """
        This function takes paths to wave file and pitchtier file 
        and runs praat script to resynthesise a new wave file.
        """
        cmd = r'bin\Praat.exe --run bin\wav_and_pitchtier_to_wav.psc "' + os.path.realpath(wave) + '" "' + os.path.realpath(pitchtier) +'"'
        os.system(cmd)

    def clean_up(self):
        """
        This function will clean up all unneccesary files in app folder as well as 
        moving new wave file to desired location.
        """

        os.remove('syc.txt')
        os.remove('text_and_fujifeat.csv')
        os.remove('PAC_log.txt')
        os.remove('temp_resynthesised.PitchTier')
        sound_file_name = self.sound_file.split('\\')[1].split('.')[0]
        [os.remove(sound_file_name + '.' + i) for i in ['csv', 'lab', 'PAC', 'Pitch']]

        # Move new wave file from app folder to app/sound
        new_wav_file = 'new_' + sound_file_name + '.wav'
        shutil.move(new_wav_file, os.path.join('sound', new_wav_file))

if __name__ == '__main__':

    """
    Right now we can only resynthesise for sentense water harms the new born boy.
    """
    
    tts = TTSCreator('Water harms the new born boy', 'sad')
    tts.create_speech()