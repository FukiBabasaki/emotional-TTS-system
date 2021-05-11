from sound_creation.sound_creator import SoundCreator
from fujisaki_extractor.fujisaki_extractor import FujisakiExtractor
from feature_extraction.feature_extractor import FeatureExtractor
from resynthesis.resynthesiser import Resynthesiser
import os

class TTSCreator:
    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion
        self.sound_creator = SoundCreator(text)
        self.fujisaki_extractor = FujisakiExtractor(text)
        self.feature_extractor = FeatureExtractor(text)
        self.resynthesiser = Resynthesiser(emotion)


    def create_speech(self):

        sound_file = self.sound_creator.create_sound()
        fujisaki_features = self.fujisaki_extractor.extract_features(sound_file)
        text_features = self.feature_extractor.extract_text_features()
        merged_csv = self.__merge_textfeat_and_fujifeat(fujisaki_features, text_features)
        pitchtier = self.resynthesiser.resynthesis(merged_csv, self.fujisaki_extractor.pac, self.fujisaki_extractor.lab)

        self.__generate_wav_from_pitchtier(sound_file, pitchtier)
        
        return "input: " + self.text + " " + self.emotion

    def __merge_textfeat_and_fujifeat(self, fuji, textfeat):
        import pandas as pd

        dfs = [pd.read_csv(fuji), pd.read_csv(textfeat)]
        df_merged = pd.concat(dfs, axis=1)
        df_merged.to_csv('text_and_fujifeat.csv')

        return os.path.realpath('text_and_fujifeat.csv')

    def __generate_wav_from_pitchtier(self, wave, pitchtier):
        cmd = r'bin\Praat.exe --run bin\wav_and_pitchtier_to_wav.psc ' + wave + ' ' + pitchtier
        os.system(cmd)

if __name__ == '__main__':
    tts = TTSCreator('Water harms the new born boy', 'sad')
    tts.create_speech()