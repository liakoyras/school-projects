module arbiter(
	input logic [2:0] lowp_i,
	input logic [7:0] reqs_i,
	output logic any_grant_o,
	output logic [2:0] grants_o,
	output logic [3:0] cnt_o
);

assign any_grant_o = &(grants_o);

always_comb begin
	cnt_o = 0;

    for(int i=0; i<7; i=i+1) begin
        if(reqs_i[i]) begin
            cnt_o = cnt_o + 1;
		end	
	end
end

logic any_request;
assign any_request = &reqs_i;

logic found;
int index;

always_comb begin
	/* if(any_request) begin
		found = 0;
		grants_o = 0;
		for(int j = lowp_i + 1; j <= lowp_i + 1 + 7; j=j+1) begin
			if(reqs_i[j] && (!found)) begin
				grants_o = j;
				found = 1;
			end
		end
	end else begin
		grants_o = 8'b00000000;
	end */
	
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