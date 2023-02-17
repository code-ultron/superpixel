import numpy as np

def analyze_mask(mask, CONVERSION_FACTOR, precision=3):
    
    # Find Wear
    flank_wear_widths = []
    for x_pos in np.array(range(0, mask.shape[1])):
        white_pixel = np.where((mask[:,x_pos:x_pos+1,:] == [255, 255, 255]).all(axis=2))
        flank_wear_widths.append(len(white_pixel[0]))

        
    # Only consider columns with at least 5 pixel of wear
    flank_wear_widths = [w for w in flank_wear_widths if w > 5]
    if len(flank_wear_widths) == 0:
        flank_wear_widths.append(0)
    
    
    vb_avg = np.round(np.mean(flank_wear_widths) * CONVERSION_FACTOR, precision)
    vb_median = np.round(np.median(flank_wear_widths) * CONVERSION_FACTOR, precision) 
    vb_max = np.round(np.max(flank_wear_widths) * CONVERSION_FACTOR, precision) 
    vb_percentile = np.round(np.percentile(flank_wear_widths,95) * CONVERSION_FACTOR, precision)
    
    return {"vb_avg [mm]": vb_avg,
           "vb_median [mm]": vb_median,
           "vb_max [mm] ": vb_max,
           "vb_percentile [mm]": vb_percentile}
