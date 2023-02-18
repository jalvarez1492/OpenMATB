from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
from WaveletFlatGaussian import WaveletFlatGaussian_Conv


EI = 5

def generateEI():
    global EI 

    # generate filters
    
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    
    inlet.flush()
    
    time_diff = []
    mean = 0
    
    while True:
        #print('Generating EI...')
        chunk, timestamps = inlet.pull_chunk()
        if timestamps:
            
            # Calculate outputs from filter bank
            for thisChannel in range(np.shape(chunk)[1]):
                print(thisChannel)
                # print("Chunck: ",chunk,"\n")
                # print(np.shape(chunk))
                
                chunkArray = np.array(chunk)
                chunkArray = chunkArray.T
            
                
                FilterOuts = WaveletFlatGaussian_Conv(chunkArray[thisChannel, :], 256, 60.0)
                
                # Group outputs from filters according to their frequencies
                alpha= FilterOuts[3,:] + FilterOuts[2,:]
                beta = FilterOuts[4,:] + FilterOuts[5,:] + FilterOuts[6,:] + FilterOuts[7,:]
                theta = FilterOuts[1,:]

                # Run calculations for engagement
                engagement = np.sum(beta)/(np.sum(alpha)+np.sum(theta))
                eng_idx = beta/(alpha+theta)
                # eng_idx_sum= np.sum(eng_idx)
                # eng_idx_mean = np.mean(eng_idx)
                # eng_idx_std = np.std(eng_idx)

                # Print outputs
                print("For channel ",thisChannel, " native engagement is ", engagement)
                #print("For channel ",i, "native  eng_idx_sum is ", eng_idx_sum1, "\n")
                #print("For channel ",i, "native  eng_idx_mean is ", eng_idx_mean1, "\n")
                #print("For channel ",i, "native  eng_idx_std is ", eng_idx_std1, "\n")
                EI = engagement                                                                      # Placeholder
       
            #print(timestamps)
            print("The current time is ",time.time())
            latency = time.time() - timestamps[len(timestamps) - 1]
            #print("Latency: ", latency)
            time_diff.append(latency)
            mean = np.mean(time_diff)
            print("Mean free time: ",mean)
            print("Samples pulled: ", np.shape(chunk)[0])
            
        time.sleep(1)
