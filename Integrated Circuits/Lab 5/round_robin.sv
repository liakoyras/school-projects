module round_robin(
	input logic clock,
	input logic reset,
	input logic [7:0] requests,
	output logic [6:0] low_priority_out,
	output logic [6:0] requests_count,
	output logic [7:0] grants_8
);

logic [2:0] low_priority;
logic [2:0] grants_3;
logic [3:0] number_requests;
logic enable;

arbiter a1 (
	.lowp_i(low_priority),
	.reqs_i(requests),
	.grants_o(grants_3),
	.any_grant_o(enable),
	.cnt_o(number_requests)
);

priority_register p1 (
	.clock(clock),
	.reset(reset),
	.d_i(grants_3),
	.en_i(enable),
	.q_o(low_priority)
);

m_7_segment_decoder d1 (
	.in_3_bits(low_priority),
	.out_7_bits(low_priority_out)
);

m_7_segment_decoder d2 (
	.in_3_bits(number_requests),
	.out_7_bits(requests_count)
);

m_3_to_8_decoder d3 (
	.in_3_bits(grants_3),
	.enable(enable),
	.out_8_bits(grants_8)
);

endmodule