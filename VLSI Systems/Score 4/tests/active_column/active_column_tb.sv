module active_column_tb;

logic [6:0] play;
logic [2:0] column;

active_column col (.play(play), .column(column));

initial begin
	play = 7'b0000001;
	#5ns
	play = 7'b0000010;
	#5ns
	play = 7'b0000100;
	#5ns
	play = 7'b0000010;
end


endmodule