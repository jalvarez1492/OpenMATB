from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
from WaveletFlatGaussian import WaveletFlatGaussian_Conv
import pickle
import os

# Init global vars
EI = 0
lastEI = 0

def generateEI():
    # global vars to communicate with OpenMATB
    global EI
    global lastEI 
    
    # Setup pickle file for post processing data
    pkl_path = r'C:\Users\Dr. Nicholas Napoli\Desktop\Senior\SeniorDesign\Code\OpenMATB\PickleFilesGoHere\HereActually'
    filename = str(time.time()) +  '.pickle'

    # Setup streams and inlets for LSL
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    
    # Latency testing
    #time_diff = []
    #timeMean = 0
    
    # Flush the stream of data beforehand
    inlet.flush()
    # Repeat until OpenMATB ends
    while True:
        chunk, timeStamps = inlet.pull_chunk()
        if timeStamps:
            # Create empty lists to store data
            channelsEngagement = [] 
            channelsEngIdx = [] 
            channelsAlpha = []
            channelsBeta = []
            channelsTheta = []
            # Calculate outputs from filter bank
            for thisChannel in range((np.shape(chunk)[1])-1): # discard the aux channel according to paper           # uncomment me for all channels
            #for thisChannel in range(1):                                                                            # uncomment me for one channel
                #print(thisChannel)
                # print("Chunck: ",chunk,"\n")
                # print(np.shape(chunk))
                
                # Reformat Muse data to be easier to work with
                chunkArray = np.array(chunk)
                chunkArray = chunkArray.T
            
                # Filter data using wavelet script
                filteredOuts = WaveletFlatGaussian_Conv(chunkArray[thisChannel, :], 256, 60.0)
                
                # Group outputs from filters according to their frequencies
                alpha= filteredOuts[3,:] + filteredOuts[2,:]
                beta = filteredOuts[4,:] + filteredOuts[5,:] + filteredOuts[6,:] + filteredOuts[7,:]
                theta = filteredOuts[1,:]

                # Run calculations for engagement
                thisChannelEngagement = np.sum(beta)/(np.sum(alpha)+np.sum(theta))
                thisChannelEngIdx = beta/(alpha+theta)
                #print("This channel engagement",thisChannelEngagement)
                #print("This eng_idx is lenght ", np.size(thisChannelEngIdx))
                #print("Channel ",thisChannel, "engagement is ",thisChannelEngagement)
                # eng_idx_sum= np.sum(eng_idx)
                # eng_idx_mean = np.mean(eng_idx)
                # eng_idx_std = np.std(eng_idx)

                # Print outputs
                #print("For channel ",thisChannel, " native engagement is ", thisChannelEngagement)
                #print("For channel ",i, "native  eng_idx_sum is ", eng_idx_sum, "\n")
                #print("For channel ",i, "native  eng_idx_mean is ", eng_idx_mean, "\n")
                #print("For channel ",i, "native  eng_idx_std is ", eng_idx_std, "\n")
                
                # Append this channel's engagement to rest of channels in this timestamp chunk
                channelsEngagement.append(thisChannelEngagement)                                       
                channelsEngIdx.append(thisChannelEngIdx)
                channelsAlpha.append(alpha)
                channelsBeta.append(beta)
                channelsTheta.append(theta)
            
            # Find the mean engagement to change automatic/manual control
            EI = np.mean(channelsEngagement)
            print("The mean engagement is ",EI)
            #print("ChannelEngIdx size: ", np.shape(channelsEngIdx)) 
            #print("The last engagement was ",lastEI)
            
            # Save the data for post processing
            
            # On the first iteration, the pickle file won't exist. On all others,
            # we need to first load the current pickle then append all new data
            if os.path.exists(pkl_path + filename):
                # Pickle file already exists
                
                # Load the current pickle
                
                # Sync the pickle with new data
                pass
            else: 
                # Pickle file doesn't exist yet so we just create one and write to it
                with open(pkl_path + filename, 'wb') as f:
                    pickle.dump({'timeStamps': timeStamps,'ChannelEngagements':channelsEngIdx,'ChannelAlphas': channelsAlpha,'ChannelBetas': channelsBeta,'ChannelThetas':channelsTheta, }, f)
            
            # Latency debugging
            #print(timeStamps)
            #print("The current time is ",time.time())
            #latency = time.time() - timeStamps[len(timeStamps) - 1]
            #print("Latency: ", latency)
            #time_diff.append(latency)
            #timeMean = np.mean(time_diff)
            #print("Mean free time: ",timeMean)
            #print("Samples pulled: ", np.shape(chunk)[0])
        
        # Wait time between swtiching between automatic and manual control using pos/neg feedback
        time.sleep(2)
        lastEI = EI 
