from sound_creation.sound_creator import SoundCreator
from fujisaki_extractor.fujisaki_extractor import FujisakiExtractor
from feature_extraction.feature_extractor import FeatureExtractor
from resynthesis.resynthesiser import Resynthesiser

class TTSCreator:
    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion
        self.sound_creator = SoundCreator(text)
        self.fujisaki_extractor = FujisakiExtractor(text)
        self.feature_extractor = FeatureExtractor(text)
        self.resynthesiser = Resynthesiser()


    def create_speech(self):

        sound_file = self.sound_creator.create_sound()
        '''
        # TO DO
        fujisaki_features = self.fujisaki_extractor.extract_features(sound_file)
        text_features = self.feature_extractor.extract_features(self.text)
        speech_with_emotion = self.resynthesiser.resynthesise()
        '''

        return "input: " + self.text + " " + self.emotion

    def __merge_textfeat_and_fujifeat(self, fuji, textfeat):
        import pandas as pd
        dfs = [pd.read_csv(fuji), pd.read_csv(textfeat)]
        df_merged = pd.concat(dfs, ignore_index=True, axis=1)
        df_merged.to_csv('text_and_fujifeat.csv')