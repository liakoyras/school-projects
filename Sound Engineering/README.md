# Sound Engineering
The MATLAB projects I contributed in for my 5th semester Sound Engineering course.<br>
These were group projects. The members of the group were:
- Ilias Chanis
- Konstantina Dimitriadou
- Evaggelia Nivolianiti
- Fotini Anna Trikou

More information about the projects:

## Project 1 (Short-time Fourier transform applications)
We created functions that break the sound into frames and can then reconstruct it.<br>
Then, we tested if this process works and created spectrograms.<br>
 After this, we calculated if each frame is considered voiced or unvoiced and used this distinction in order to:
- Create a de-esser (a program which reduces the volume of "s" or "sh" sounds)
- Create a voice activity detector (VAD) that recognizes when the frame is active or not

## Project 2 (Linear Prediction Coefficients)
We used some of the routines created for the previous project in order to calculate the LPC coefficients and use them to:
- Create a basic voice synthesizer that recreates human voice
- Create a sound compression and decompression system
- Calculate the effect of the LPC class on the error
- Create a system to prove that the majority of the information is contained within the LPC coefficients and not on the error by morphing the error of person A to sound like person B using B's LPC coefficients

## Project 3 (Sound effects)
We created filters that produce many sound effects, including fuzz, a general nonlinear filter, chorus and reverb (for different rooms whose impulse responses are in the file rooms.mat).<br>
We also wrote code for testing the effect of aplying more than one filter in different orders.
