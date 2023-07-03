from tbparse import SummaryReader
import pandas as pd
import argparse



def parse_tb2(logdir, csv_out):
	reader = SummaryReader(logdir, extra_columns=set(['wall_time']))
	df = reader.scalars

	df['wall_clock'] = pd.to_datetime(df.wall_time, unit="s")
	first = df.wall_time[0]
	df['runtime_to_report'] = df.wall_time - first
	
	if csv_out:
		df[df['tag'] == 'Train/Samples/train_loss'].to_csv(csv_out)


	return df[df['tag'] == 'Train/Samples/train_loss']


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parse TensorBoard Event files and prints them')
	parser.add_argument('logdir', help='TensorBoard logdir')
	parser.add_argument('--csv_out', help='csv output file', required=False, default=None)
	args = parser.parse_args()

	print(parse_tb2(args.logdir, args.csv_out))
