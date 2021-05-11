import requests, os, csv
class FeatureExtractor:
    def __init__(self, text):
        self.text = text
        self.cleaned_text = self.text.replace(' ', '%20')
    

    def extract_text_features(self):
        os.chdir('feature_extraction')
        # this is temporary command to get features.
        # It needs to be replaced with proper feature extractor for Eng-NZ
        # Right now the features are related to US English
        # One possible solution: http://robotspeech.auckland.ac.nz:59125/process?INPUT_TEXT=%T%&INPUT_TYPE=TEXT&OUTPUT_TYPE=TARGETFEATURES&LOCALE=en_NZ
        
        r = requests.get(r"http://mary.dfki.de:59125/process?INPUT_TEXT=" + self.cleaned_text + r"&INPUT_TYPE=TEXT&OUTPUT_TYPE=TARGETFEATURES&LOCALE=en_US")
        new_csv = self.__textfeatures_to_csv(r.text)
        os.chdir('..')
        return new_csv
    
    def __textfeatures_to_csv(self, textfeature):
        
        # First separate the text by new line
        lines = textfeature.split('\n\n')
        print(lines[2])
        with open('phone_and_text_features.csv', 'w', newline='') as csv_file:
            # choose lines that are relevant to features
            features = []
            for i in range(1,107):
                features.append(lines[0].split('\n')[i].split()[0])
            
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(features)

            for line in lines[2].split('\n'):
                if line == '\n':
                    # we interated over all phones
                    break
                else:
                    l = line.split()
                    csv_writer.writerow(l[:-3])
            
            return os.path.realpath(csv_file.name)

if __name__ == '__main__':
    fe = FeatureExtractor('hello')
    fe.extract_text_features()