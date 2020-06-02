library verilog;
use verilog.vl_types.all;
entity synchronization is
    port(
        clk             : in     vl_logic;
        rst             : in     vl_logic;
        hsync           : out    vl_logic;
        vsync           : out    vl_logic;
        rows            : out    vl_logic_vector(9 downto 0);
        columns         : out    vl_logic_vector(9 downto 0)
    );
end synchronization;
