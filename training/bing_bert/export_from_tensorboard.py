from tbparse import SummaryReader
import pandas as pd
import argparse



def parse_tb2(logdir, csv_out, threshold, min_iter):
	reader = SummaryReader(logdir, extra_columns=set(['wall_time']))
	df = reader.scalars

	df['wall_clock'] = pd.to_datetime(df.wall_time, unit="s")
	first = df.wall_time[0]
	df['Runtime'] = df.wall_time - first
	
	loss_df = df[df['tag'] == 'Train/Samples/train_loss'].copy()
	loss_df.rename(columns = {'step' : 'Samples', 'value' : 'Loss'}, inplace=True)
	loss_df.drop(columns = ['tag', 'wall_time', 'wall_clock'], inplace=True)
	
	target_runtime_idx = find_runtime(loss_df, threshold, min_iter)
	
	if csv_out:
		loss_df.to_csv(csv_out)
	
	return loss_df, target_runtime_idx

	
	
def find_runtime(df, threshold, min_iter):
	for i in df.index:
		if df['Loss'][i]  < threshold:
			val = True
			for j in range(min_iter+1):
				val = val and (df['Loss'][i+j] < threshold)
			if val:
				return i
		if i == df.index[-min_iter-1]:
			return False



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parse TensorBoard Event files and prints them')
	parser.add_argument('logdir', help='TensorBoard logdir')
	parser.add_argument('--csv_out', help='csv output file', default=None)
	parser.add_argument('--threshold', help='threadhold values are acceptable', default=8.5, type=float)
	parser.add_argument('--min_iter', help='minimum iteration under the threshold', default=2, type=int)

	args = parser.parse_args()

	df, target_runtime_idx = parse_tb2(args.logdir, args.csv_out, args.threshold, args.min_iter)
	print(df.to_string())
	print('='*60)
	if target_runtime_idx:
		print(f"Time to reach target loss: {df['Runtime'][target_runtime_idx]:.2f} s")
	else:
		print("Target loss not reached")
