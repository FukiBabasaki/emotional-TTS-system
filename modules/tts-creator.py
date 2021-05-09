
class TTSCreator:
    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion
        self.sound_creator = SoundCreator(text)
        self.fujisaki_extractor = FujisakiExtractor(text)
        self.feature_extractor = FeatureExtractor(text)
        self.resynthesiser = Resynthesiser()


    def create_speech():
        sound_file = self.sound_creator.create_sound()
        fujisaki_features = self.fujisaki_extractor.extract_features()
        text_features = self.feature_extractor.extract_features()
        speech_with_emotion = self.resynthesiser.resynthesise()

        return speech_with_emotion