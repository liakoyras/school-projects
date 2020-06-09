module active_column (
	input logic [6:0] play,
	
	output logic [2:0] column
);

// lookup table to find the selected column
always_comb begin
	if(play==1) column = 0;
	else if(play==2)  column = 1;
	else if(play==4)  column = 2;
	else if(play==8)  column = 3;
	else if(play==16) column = 4;
	else if(play==32) column = 5;
	else if(play==64) column = 6;
	else column = -1;
end


endmodule