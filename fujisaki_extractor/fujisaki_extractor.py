import requests, os

DEBUG_MODE = False
class FujisakiExtractor:
    NUL = ''
    if DEBUG_MODE:
        NUL+=' > NUL'
    
    def __init__(self, text):
        self.text = text

    def extract_features(self, sound):
        self.sound = sound
        text_grid = self.__generate_textgrid()
        f0_ascii = self.__generate_f0_file(sound, text_grid)
        pac = self.__generate_PAC(f0_ascii)
        self.__generate_csv(pac)

    def __generate_textgrid(self):
        file = open('text.txt', 'w+')
        file.write(self.text)
        file.close()

        cmd = "curl --silent -v -X POST -H 'content-type:multipart/form-data' -F SIGNAL=@" + self.sound + " -F LANGUAGE=eng-NZ -F TEXT=@"+ os.path.realpath(file.name) + " https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic" + " > downloadpath.txt" + self.NUL
        os.system(cmd)

        os.remove(file.name)

        f = open('downloadpath.txt', 'r')
        text = f.read()
        f.close()
        os.remove(f.name)
        download_link = text.split('<downloadLink>')[1].split('</downloadLink>')[0]

        r = requests.get(download_link)
        with open('speech.TextGrid', 'w') as f:
            f.write(r.text)
        
            return os.path.realpath(f.name)
        
    def __generate_f0_file(self, sound, text_grid):
        # create lab file from TextGrid file.
        cmd = r'bin\textgrid2lab.exe ' + text_grid + ' 3' + self.NUL
        os.system(cmd)
        
        # Create pitch file from praat.exe
        pitch = text_grid.replace('.TextGrid', '.Pitch')
        cmd = r'..\bin\Praat.exe --run bin\soundToPitch.psc ' + sound + " " + pitch + self.NUL
        os.system(cmd)

        # need to replace frames with frame to avoid conflicts
        with open(pitch, 'r') as f:
            old_p = f.read()

        with open(pitch, 'w') as f:
            new_p = old_p.replace('frames', 'frame').replace('candidates', 'candidate')
            f.write(new_p)

        # finally conver pitch file to f0_ascii file
        cmd = r'bin\pitch2f0_ascii.exe ' + pitch + self.NUL
        os.system(cmd)

        return os.path.realpath(pitch.replace('Pitch', 'f0_ascii'))

    def __generate_PAC(self, f0_ascii):
        cmd = 'bin\interpolation ' + f0_ascii + " 0 4 0.0001 auto 3" + self.NUL
        os.system(cmd)
        os.remove('interpolation.txt')
        os.remove('speech.4.txt')

        return f0_ascii.replace('f0_ascii', 'PAC')

    def __generate_csv(self, pac):
        import glob
        import csv

        #labnpac.exe only works with a txt file containing 
        #list of PAC files.
        with open('list_pac.txt', 'w') as f:
            f.write(pac)
        
        cmd = 'bin\labnpac.exe ' + f.name + self.NUL
        os.system(cmd)

        os.remove(f.name)

        # Transform the syc.txt file into csv file
        with open(pac.replace('PAC', 'lab'), 'r') as lab:
            reader = csv.reader(lab)
            rows = []
            for row in reader:
                rows.append(row[0])
            DUR = rows[-1].split()[0]

        with open(r'speech.csv', 'w', newline='') as new_csv:
            writer = csv.writer(new_csv)
            writer.writerow(['file', 'smpa', 'abs_aa_syn', 'dur_acc_syn', 'fmin_syn', 'switch_syn', 'ap_syn', 'dur_phr_syn', 'DUR'])

            ap_prev = '0'
            dur_phr_prev = '0'
            with open(r'syc.txt', 'r') as inf:
                next(inf)
                reader = csv.reader(inf, delimiter = '\t')
            
                for row in reader:
                    file = row[0]
                    smpa = row[1]
                    aa = row[4]
                    t1 = row[5]
                    t2 = row[6]
                    fmin = row[10]
                    switch = row[17]

                    ap = row[8]
                    # if it is 0, copy the previous value
                    if float(ap) == float(0):
                        ap = ap_prev
                    else :
                        ap_prev = ap

                    dur_phr = row[9]
                    if float(dur_phr) == float(0):
                        dur_phr = dur_phr_prev
                    else :
                        dur_phr_prev = dur_phr
                    
                    writer.writerow([file, smpa, aa, float(t2) - float(t1), fmin, switch, ap, dur_phr, DUR])
        
        os.remove('syc.txt')
        return pac.replace('PAC', 'csv')

if __name__ == '__main__':
    fe = FujisakiExtractor('hello')
    fe.extract_features(os.path.abspath(r"C:\Users\fbab9\OneDrive\Documents\UoA\Summer_Research\emotional-TTS-system\sound_creation\speech.wav"))