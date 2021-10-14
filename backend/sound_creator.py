import requests, os
class SoundCreator:
    def __init__(self, text):
        self.text = text
        
    def create_sound(self):
        '''
        This function saves a wav file that contains 
        speech of the input text. 
        Returns the path to the wav file.
        '''

        # HTTP GET wave file from link to aotearoa voices.
        
        ### TOO manu retries throws error and gets rejected by the server.
        # url = "https://aotearoavoices.nz/api/requestAudio?text=" + self.text + "&engine=MaryTTS&voice=Male+NZ-English+1&fileType=.wav"
        # r = requests.get(url)
        # with open(os.path.join('sound', 'speech.wav'), 'wb') as wav_file:
        #     wav_file.write(r.content)
        
        # return os.path.relpath(wav_file.name)

        return os.path.join('sound', 'speech.wav')
