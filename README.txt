Project for evaluating adversarial opponents under an expectation-maximization algorithm framework
	The Expectation step calculates the optimal latent team ratings given the model parameters
	The Maximization step calculates the optimal model parameters given the latent team ratings
	Extensions for prior distributions over latent ratings and weighting training examples (useful for neutral site games) present

The generalized_single_season_main.py is the high level code for running the algorithm on any dataset containing home and away teams with home and away scores
This has been tested on:
	NBA data
	Men's College Lacrosse data
	Men's College Basketball data (produces similar ratings to kenpom.com when using prior distribution)