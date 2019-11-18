module winX (
input logic [8:0] X,
output logic win
);

always_comb
begin
	if(X[4]) begin
		if((X[0] && X[8]) || (X[1] && X[7]) || (X[2] && X[6]) || (X[3] && X[5]))
			win = 1;
		else
			win = 0;
	end
	else begin
		if(X[0]) begin
			if((X[1] && X[2]) || (X[3] && X[6]))
				win = 1;
			else
				win = 0;
		end
		else begin
			if(X[8]) begin
				if((X[2] && X[5]) || (X[6] && X[7]))
					win = 1;
				else
					win = 0;
			end
			else
				win = 0;
		end
	end			
end
endmodule
