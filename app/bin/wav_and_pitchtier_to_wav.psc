#This script merges wav and pitchtier into one wav file

form Input wave file
	sentence wave "haha"
	sentence pitchtier "haha"
endform

#use tier number 1
pitch_file = Read from file... 'pitchtier$'
	
wav_file = Read from file... 'wave$'

#Create manipulation object
select wav_file
manipulation = To Manipulation... 0.01 75.0 600.0

plusObject: pitch_file

#Replace pitch tier
Replace pitch tier

select manipulation
Get resynthesis (overlap-add)

Save as WAV file... ..\new_speech.wav
