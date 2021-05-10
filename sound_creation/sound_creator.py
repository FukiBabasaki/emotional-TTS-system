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
        url = "https://aotearoavoices.nz/api/requestAudio?text=" + self.text + "&engine=MaryTTS&voice=Male+NZ-English+1&fileType=.wav"
        r = requests.get(url)

        wav_file = open('speech.wav', 'wb')
        wav_file.write(r.content)

        wav_file.close()
        return os.path.realpath(wav_file.name)

if __name__ == '__main__':
    sc = SoundCreator('hello')
    name = sc.create_sound()

    print(name)