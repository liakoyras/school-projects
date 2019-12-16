module arbiter(
	input logic [2:0] lowp_i,
	input logic [7:0] reqs_i,
	output logic any_grant_o,
	output logic [2:0] grants_o,
	output logic [3:0] cnt_o
)

assign any_grant_o = &(grants_o);

always_comb begin
	cnt_o = 0;

    for(int i=0; i<7; i=i+1) begin
        if(reqs_i[i]) begin
            cnt_o = cnt_o + 1;
		end	
	end
end

always_comb begin
	if(&reqs_i) begin
		for(int j = lowp_i + 1; j <= lowp_i + 1 + 7; j=j+1) begin
			if(reqs_i[j]):
				grants_o[j] = j;
		end
	end else begin
		grants_o = 0;
	end
end