import os
import pandas as pd

class Resynthesiser:
    """
    A class that is responsible for predicting and creating new pitchtier from predicted fujisaki features.
    """
    def __init__(self, emotion):
        self.emotion = emotion

    def resynthesis(self, features, pac, lab):
        """
        This function will call predict function to create new pitchtier.
        """
        self.features = features
        self.pac = pac
        self.lab = lab

        predicted = self.predict()
        new_pac = self.pac_from_prediction(predicted)
        pichtier = os.path.relpath(self.pac_to_pitchtier(new_pac))
        
        os.remove(new_pac)
        os.remove(predicted)

        return pichtier
    
    def predict(self):
        ## We don't predict for now. 
        ## We already have predicted csv file for water harms new born boy

        # from ml.predict import Predictor
        # predictor = Predictor(self.features, self.emotion)
        # predicted = predictor.predict()
        predicted = os.path.join('data', 'predicted.csv')
        return predicted

    def pac_from_prediction(self, prediction):
        dataset = pd.read_csv(prediction)
        with open(self.pac, 'r') as PAC:
            lines=PAC.readlines() #to read lines according to line number
            num_lines = len(lines)    
            num_of_accents = len(dataset) 
            num_of_phrases = [int(i) for i in lines[7].split()]
            num_of_phrases = pd.DataFrame(num_of_phrases)
            num_of_phrases = int(num_of_phrases[0][0])
            lines[9] = str(dataset.iloc[0]['fmin_predicted']) + '\n'
            lines[8] = str(num_of_accents) + '\n'
        
        for i in range(0, num_of_accents + num_of_phrases):
            lines.append('\n')
    
        #for ap i=0
        for i in range(0, num_of_phrases ):
            ap_details = str(lines[20+i])
            start_time = ap_details.split("  ")[0]
            
            t0 = start_time
            row = dataset.iloc[(dataset['ton']-float(t0)).abs().argsort()[:1]] #row closest to the phgrase command start
        
        #now need to get ap and duration of phrase from this row
        ap = row.iloc[0]['ap_predicted']
        dur_phr = row.iloc[0]['dur_phr_predicted']
        
        t01 = float(t0) + dur_phr
        ap_string = str(t0) + '  ' +str(t01) + '  ' + str(ap) + '  ' + '3.00' + '\n'
        lines[19 + i] = ap_string
        #for aa
        for i in range(0, num_of_accents ):
            aa = dataset.iloc[i]['aa_predicted']
            t1 = dataset.iloc[i]['ton']
            t2 = dataset.iloc[i]['toff']
            aa_string = str(t1) + '  ' + str(t2) + '  '  + str(aa) + '  ' +  '20.00' + '\n'
            lines[19 + num_of_phrases + i] = aa_string
        
        syn_pac = 'resynthesised.PAC'
        with open(syn_pac, 'w') as f:
            for i in range(0,len(lines)):
                f.write(str(lines[i]))

        return syn_pac
    
    def pac_to_pitchtier(self, pac):
        with open(self.lab, 'r') as label:
            lines = label.readlines()
        duration = lines[-1].split(' ')[0]
        cmd = r'bin\pac2pitchtier.exe ' + pac + ' 1 0 ' + duration + ' temp'
        os.system(cmd)

        pitchtier_name = "temp_"+ pac.replace('PAC', 'PitchTier')
        return pitchtier_name
