**MUST READ**
1. paste trained_model28.pkl into the folder "result_0522"
2. paste trained_model56.pkl into the folder "result2_0522"
3. paste CNN.pkl and CNN56.pkl into the folder "result_cnn_0522"
4. run CapsuleNet2.ipynb to show the testing results of all models
5. if want to train models, 
	a. for CapsNet28
	-	change the argument of weights of parser from 
		default=ROOT+'/result_0522' + '/trained_model28.pkl' to default=None
	- 	remove '-t' from args = parser.parse_args(args=['-t']) to be
		args = parser.parse_args(args=[])
	b. for CapsNet56
	-	change the argument of weights of parser2 from 
		default=ROOT+'/result2_0522' + '/trained_model56.pkl' to default=None
	- 	remove '-t' from args2 = parser2.parse_args(args=['-t']) to be
		args2 = parser2.parse_args(args=[])
	c. for CNN28
	-	change testing_cnn = True to testing_cnn = False
	d. for CNN56
	-	change testing_cnn56 = True to testing_cnn56 = False
