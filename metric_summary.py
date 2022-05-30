import os
import sys
import pandas as pd

def summarize_metrics(base_dir):
    """
    Goes through all subfolders of base_dir and looks for files named 'metrics.csv'
    For each one of those files, calculate the min, max, mean, and std of the columns psnr, ssim, and lpips.
    Save this information to a new file in the same directory called 'summary.csv'
    """
    psnr_means = {}
    ssim_means = {}
    lpips_means = {}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == 'metrics.csv':
                # Summarize the metrics using describe()
                metrics_df = pd.read_csv(os.path.join(root, f))
                description = metrics_df.describe()
                description.to_csv(os.path.join(root, 'summary.csv'), index=True)
                # Get the last directory in the path
                val_name = root.split('/')[-1]
                exp_name = root.split('/')[-4]
                # Add the summary to the means
                if val_name not in psnr_means:
                    psnr_means[val_name] = {}
                    ssim_means[val_name] = {}
                    lpips_means[val_name] = {}
                psnr_means[val_name][exp_name] = description['psnr']['mean']
                ssim_means[val_name][exp_name] = description['ssim']['mean']
                lpips_means[val_name][exp_name] = description['lpips']['mean']
    
    # Save the overall_means to a file
    pd.DataFrame(psnr_means).to_csv(os.path.join(base_dir, 'psnr_means.csv'), index=True)
    pd.DataFrame(ssim_means).to_csv(os.path.join(base_dir, 'ssim_means.csv'), index=True)
    pd.DataFrame(lpips_means).to_csv(os.path.join(base_dir, 'lpips_means.csv'), index=True)

if __name__ == '__main__':
    # Get the base directory from the command line
    base_dir = sys.argv[1]
    summarize_metrics(base_dir)