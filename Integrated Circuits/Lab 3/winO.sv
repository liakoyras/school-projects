module winO (
input logic [8:0] O,
output logic win
);

always_comb
begin
	if(O[4]) begin
		if((O[0] && O[8]) || (O[1] && O[7]) || (O[2] && O[6]) || (O[3] && O[5]))
			win = 1;
		else
			win = 0;
	end
	else begin
		if(O[0]) begin
			if((O[1] && O[2]) || (O[3] && O[6]))
				win = 1;
			else
				win = 0;
		end
		else begin
			if(O[8]) begin
				if((O[2] && O[5]) || (O[6] && O[7]))
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