module arbiter(
	input logic [2:0] lowp_i,
	input logic [7:0] reqs_i,
	output logic any_grant_o,
	output logic [2:0] grants_o,
	output logic [3:0] cnt_o
);

assign any_grant_o = (|(reqs_i))? 1'b1 : 1'b0 ;

always_comb begin
	cnt_o = 0;

    for(int i=0; i<8; i=i+1) begin
        if(reqs_i[i]) begin
            cnt_o = cnt_o + 1;
		end	
	end
end

logic found;
logic [2:0] index;

always_comb begin	
	found = 0;
	grants_o = 0;
	index = (lowp_i + 1);
	for(int j = 0; j <  8; j = j + 1) begin
		if(reqs_i[index] && (!found)) begin
			grants_o = index;
			found = 1;
		end
		index = index + 1;
	end
end

endmodule