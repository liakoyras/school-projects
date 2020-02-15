/*----------------------------------------------------------------------------
 *----------------------------------------------------------------------------*/
#include <MKL25Z4.H>

__asm void operation_display(unsigned int op1, unsigned int op2, unsigned int op) {
        LDR   r5, =0x1FFFF500     ; Initialize the base address

        STR   r0, [r5]            ; Store the parameters into the
        STR   r1, [r5, #4]        ; specified memory addresses
        STR   r2, [r5, #8]        ;

        CMP   r2, #0x020          ; Select proper operation
        BEQ   addition            ; depending on the third argument
        CMP   r2, #0x021          ; of the function
        BEQ   subtraction         ;
        CMP   r2, #0x022          ;
        BEQ   multiplication      ;
        CMP   r2, #0x023          ;
        BEQ   division            ;
				
				BX    lr									; Exit the function if an incorrect operation is used

addition
        ADDS  r3, r0, r1          ; Add the numbers
        STR   r3, [r5, #12]       ; Store the result
				B     isolate							; Move on to encode the result
				
subtraction
        SUBS  r3, r0, r1          ; Subtract the numbers
        STR   r3, [r5, #12]       ; Store the result
				B     isolate							; Move on to encode the result
				
multiplication
				MOVS	r3, r1							;
        MULS  r3, r0, r3          ; Multiply the numbers
        STR   r3, [r5, #12]       ; Store the result
				B     isolate							; Move on to encode the result
				
division
        CMP   r1, #0              ; Check if one of the two operands
        BEQ   divzero             ; is equal to 0 and redirect
        CMP   r0, #0						  ;
        BEQ   zero							  ;
        MOVS  r3, #0              ; Initialize result counter
whileloop
        CMP   r0, r1              ; Check if the
        BLO   endloop             ; divident is greater than the divisor
        SUBS  r0, r0, r1          ; Subtracts the divisor value
        ADDS  r3, r3, #1          ; and increments counter
        B     whileloop           ; loop
endloop
        STR   r3, [r5, #12]       ; Store the result
				B     isolate							; Move on to encode the result
divzero
        STR   r1, [r5, #12]       ; Store zero if there is a division by 0
				B     isolate							; Move on to encode the result
zero
        STR   r0, [r5, #12]       ; Store zero if the divdent is 0
				B     isolate							; Move on to encode the result
				
				
isolate
				LSLS  r4, r3, #28         ; Isolate the 4 least significant bits
				LSRS  r4, r4, #28					; (the last hex digit)
				STR   r4, [r5, #60]		  	; Store the isolated number
				
				LSRS  r4, r3, #4 					; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #64]       ; Store the isolated number

				LSRS  r4, r3, #8 					; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #68]       ; Store the isolated number
				
				LSRS  r4, r3, #12 				; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #72]       ; Store the isolated number
				
				LSRS  r4, r3, #16 				; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #76]       ; Store the isolated number
				
				LSRS  r4, r3, #20 				; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #80]       ; Store the isolated number
				
				LSRS  r4, r3, #24 				; Isolate the next 4 digits
				LSLS  r4, r4, #28         ;
				LSRS  r4, r4, #28	        ;
				STR   r4, [r5, #84]       ; Store the isolated number
				
				LSRS  r4, r3, #28 				; Isolate the last 4 digits
				STR   r4, [r5, #88]       ; Store the isolated number
				
				
				MOVS  r6, #0              ; Initialize loop counter
encode	
				CMP   r6, #8  						; Check how many loops have been done
				BEQ   end									; and exits the function when over
				LSLS  r7, r6, #2					; Create the memory address
				ADDS  r7, r7, #60		  		; 
	   		LDR   r4, [r5, r7]				; Load the isolated number
				
				CMP   r4, #0x0						; Check what the hex digit is
				BEQ   seg0								; and encode it for the 7 segment
				CMP   r4, #0x1						; display
				BEQ   seg1								;
				CMP   r4, #0x2						;
				BEQ   seg2								;
				CMP   r4, #0x3						;
				BEQ   seg3								;
				CMP   r4, #0x4						;
				BEQ   seg4								;
				CMP   r4, #0x5						;
				BEQ   seg5								;
				CMP   r4, #0x6						;
				BEQ   seg6								;
				CMP   r4, #0x7						;
				BEQ   seg7								;
				CMP   r4, #0x8						;
				BEQ   seg8								;
				CMP   r4, #0x9						;
				BEQ   seg9								;
				CMP   r4, #0xA						;
				BEQ   segA								;
				CMP   r4, #0xB						;
				BEQ   segB								;
				CMP   r4, #0xC						;
				BEQ   segC								;
				CMP   r4, #0xD						;
				BEQ   segD								;
				CMP   r4, #0xE						;
				BEQ   segE								;
				CMP   r4, #0xF						;
				BEQ   segF								;
				
seg0    LDR   r4, =0x3F					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop
				
seg1    LDR   r4, =0x06					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

seg2    LDR   r4, =0x5B					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop
				
seg3    LDR   r4, =0x4F					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop				

seg4    LDR   r4, =0x66					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop				

seg5    LDR   r4, =0x6D					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

seg6    LDR   r4, =0x7F					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

seg7    LDR   r4, =0x07					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

seg8    LDR   r4, =0x7F				  	; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

seg9    LDR   r4, =0x6F					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segA    LDR   r4, =0x77					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segB    LDR   r4, =0x7C					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segC    LDR   r4, =0x39					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segD    LDR   r4, =0x5E					  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segE    LDR   r4, =0x79				  	; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

segF    LDR   r4, =0x71			 		  ; Matches digit with the code
				STR   r4, [r5, r7]				; Saves to the memory
				ADDS  r6, #1							; Increments loop counter
				B			encode							; Continues the loop

end
				BX    lr									; Exit function
}

int main()
{		
		const unsigned int a = 30;
		const unsigned int b = 7;
		// Select operation:
	  // 32: a+b
		// 33: a-b
		// 34: a*b
	  // 35: a/b (integer division)
		const unsigned int o = 35;
    operation_display(a, b, o);

	while (1)
			;
}
// *******************************ARM University Program Copyright © ARM Ltd 2013*************************************   
