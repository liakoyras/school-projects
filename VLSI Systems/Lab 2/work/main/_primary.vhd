library verilog;
use verilog.vl_types.all;
entity main is
    port(
        clk             : in     vl_logic;
        rst             : in     vl_logic;
        kClock          : in     vl_logic;
        kData           : in     vl_logic;
        hsync           : out    vl_logic;
        vsync           : out    vl_logic;
        red             : out    vl_logic_vector(3 downto 0);
        green           : out    vl_logic_vector(3 downto 0);
        blue            : out    vl_logic_vector(3 downto 0)
    );
end main;
