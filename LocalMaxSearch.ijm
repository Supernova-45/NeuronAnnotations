for (prominence = 1; prominence <= 10; prominence++) {
	run("Select None");
	run("Clear Results");
	close();
	open("/Users/alexandrakim/Desktop/BUGS2022/2P/Left_Forebrain_STD_1907029_9dpf_bigvsmall_52z_2P.tif");
	for (sigma = 0; sigma <= 5; sigma++) {
		run("Select None");
		run("Clear Results");
		run("Gaussian Blur...", "sigma="+sigma*0.4+" stack");
		for (slice = 1; slice <= 42; slice++){
			setSlice(slice);
			run("Find Maxima...", "prominence="+prominence+" output=[Point Selection]");
			run("Measure");
			run("Select None");
		}
		saveAs("Results", "/Users/alexandrakim/Desktop/BUGS2022/BUGScode/data/local_max/local_max_2P_prominence_"+prominence+"_sigma_"+sigma+".csv");
	}
}