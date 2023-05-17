/* ========================================
 *
 * Copyright YOUR COMPANY, THE YEAR
 * All Rights Reserved
 * UNPUBLISHED, LICENSED SOFTWARE.
 *
 * CONFIDENTIAL AND PROPRIETARY INFORMATION
 * WHICH IS THE PROPERTY OF your company.
 *
 * ========================================
*/
#include "project.h"
#include <stdio.h>
#include <stdlib.h>
#include "functions_self_defined.h"

//********************start of SECTION 1: function declarations****************//
// switch state
void switch_state(int n);
unsigned int int_to_int(unsigned int k);
void flick(int id_valid_puck);
void close_servo(int id_puck);
void init_servo(int id_puck);

//********************end of SECTION 1: function declarations****************//



//********************start of SECTION 2: variable initialization****************//
// switch state
int input = 0;
int valid_pucks[4] = {0, 0, 0, 0};

// movement general
const int MASTER_SPEED = 10000;  //##### increase if too slow; decrease if too fast #####//
const int KP = 420;             //##### increase if not straight; decrease if oscillate #####// (proportional gain)
const int COUNT_DIST_PUCKS_CENTER = 500;    //##### travelled distance between the pucks 56 and the center / between the pucks 78 and the center #####//
                                            //##### assume the pucks 56 and 78 are placed such that they are in same distance away from the respective capture zones (0.5cm as first design) #####//

// servo general
                                    //##### SHOULD BE +-50 AWAY FROM CLOSE INITIAL_SEVO_PWMX #####//
const int INITI_SERVO_PWM5 = 890;   //##### PWM compare value for servo 5 at servo initial state (state 1) #####//
const int INITI_SERVO_PWM6 = 970;   //##### PWM compare value for servo 6 at servo initial state (state 1) #####//
const int INITI_SERVO_PWM7 = 970;   //##### PWM compare value for servo 7 at servo initial state (state 1) #####//
const int INITI_SERVO_PWM8 = 890;   //##### PWM compare value for servo 8 at servo initial state (state 1) #####//

                                    //##### SHOULD BE +-50 AWAY FROM CORRESPONDING INITIAL_SEVO_PWMX #####//
const int CLOSE_SERVO_PWM5 = 940;   //##### PWM compare value for servo 5 to close the gate, same as servo final state #####//
const int CLOSE_SERVO_PWM6 = 920;   //##### PWM compare value for servo 6 to close the gate, same as servo final state #####//
const int CLOSE_SERVO_PWM7 = 920;   //##### PWM compare value for servo 7 to close the gate, same as servo final state #####//                        
const int CLOSE_SERVO_PWM8 = 940;   //##### PWM compare value for servo 8 to close the gate, same as servo final state #####//

                                    //##### my try is +- 25 away from CLOSE_SEVO_PWMX #####//
                                    //##### ensure the flicked puck is at around the center of the capture zone in y-coordinate #####//
const int FLICK_SERVO_PWM5 = 965;   //##### PWM compare value for servo 5 to flick puck #####//
const int FLICK_SERVO_PWM6 = 895;   //##### PWM compare value for servo 6 to flick puck #####//
const int FLICK_SERVO_PWM7 = 895;   //##### PWM compare value for servo 7 to flick puck #####//
const int FLICK_SERVO_PWM8 = 965;   //##### PWM compare value for servo 8 to flick puck #####//

const int WAIT = 10000;
const int ROD45 = 500;

// detect colour general
int compare_ready = 0;

// State 2: Move forward 56
const int COUNT_DIST_STARTBASE_56 = 2000;   //##### travelled distance between starting base to zone 5 and 6 where the pucks are under the colour sensors #####//

// State 4: Flick valid puck/s 56
const int COUNT_DIST_STATE4AAA = 1000;      //##### travelled distance to move to allow space for the back gates (56) to close #####//
const int COUNT_DIST_STATE4CCC = 500;       //##### travelled distance to move the robot from state4A to the center on the 4 collection zones #####//

//********************end of SECTION 2: variable initialization****************//



//********************start of SECTION 3: interrrupts function for switch and all the output states****************//
CY_ISR(ISR_Handler_SW1)
{
    switch_state(1);
}

CY_ISR(ISR_Handler_1)
{
    UART_1_PutString("State 1: Idle State\n");
    
}

CY_ISR(ISR_Handler_2)
{
    UART_1_PutString("State 2: Move forward 56\n");
    
    move_straight(1);
    until_target_count(COUNT_DIST_STARTBASE_56);
    stop_movement();
    CyDelay(100);
    
    switch_state(1);
}

CY_ISR(ISR_Handler_3)
{
    UART_1_PutString("State 3: Detect colour 56\n");
        
    PWM_colour_Start();
    
    Counter_colour_left_Start();
    Counter_colour_right_Start();
    
    LED_colour_left_Write(1);   // Turn on LED
    LED_colour_right_Write(1);
    S0_left_Write(1);    // scaling frequency
    S0_right_Write(1);
    S1_left_Write(1);   
    S1_right_Write(1);
    CyDelay(50);
    
    compare_ready = 0;
    
    char str_colour[100];
    
    int red_freq_left = 0;
    int blue_freq_left = 0;
    int green_freq_left = 0;  
    int puck_colour_left = 0;
    
    int red_freq_right = 0;
    int blue_freq_right = 0;
    int green_freq_right = 0;
    int puck_colour_right = 0;
    
    for (int i = 0; i < 3; i++)
    {
        colour_sense(1, &red_freq_left, &red_freq_right);
        colour_sense(2, &blue_freq_left, &blue_freq_right);
        colour_sense(3, &green_freq_left, &green_freq_right);
    }
    
    puck_colour_left = colour_determine(red_freq_left/3,blue_freq_left/3,green_freq_left/3);
    puck_colour_right = colour_determine(red_freq_right/3,blue_freq_right/3,green_freq_right/3);
    
    valid_pucks[0] = (puck_colour_right == 2) ? 1 : 0;
    valid_pucks[1] = (puck_colour_left == 3) ? 1 : 0;
        
    sprintf(str_colour, "puck colour 6 (left) = %d (%d)\tpuck colour 5 (right) = %d (%d)\n", puck_colour_left, valid_pucks[1], puck_colour_right, valid_pucks[0]);
    UART_1_PutString(str_colour);
    
    LED_colour_left_Write(0);
    LED_colour_right_Write(0);
    
    PWM_colour_Stop();
    PWM_colour_Init();
    
    Counter_colour_left_Stop();    
    Counter_colour_left_Init();
    Counter_colour_right_Stop();
    Counter_colour_right_Init();
    
    if (valid_pucks[0] | valid_pucks[1]) { switch_state(1); }
    else { switch_state(3); }  
    
}

CY_ISR(ISR_Handler_4)
{
    UART_1_PutString("State 4: Flick valid puck/s 56\n");
    
    move_straight(1);
    until_target_count(COUNT_DIST_PUCKS_CENTER);
    stop_movement();
    
    PWM_servo56_Start();
    
    if (valid_pucks[0] == 1){ 
        close_servo(5);
        flick(5); 
        init_servo(5);
        CyDelay(WAIT-3*ROD45);
    }
    if (valid_pucks[1] == 1){ 
        close_servo(6);
        flick(6); 
        init_servo(6);
        CyDelay(WAIT-3*ROD45);
    }
    
    PWM_servo56_Stop();

    
    switch_state(1);
    
}

CY_ISR(ISR_Handler_5)
{
    UART_1_PutString("State 5: Move forward 78A\n");
            
    move_straight(1);
    until_target_count(COUNT_DIST_PUCKS_CENTER);
    stop_movement();
    
    switch_state(2);
    
}

CY_ISR(ISR_Handler_6)
{
    UART_1_PutString("State 6: Move forward 78B\n");
    
    move_straight(1);
    until_target_count(2*COUNT_DIST_PUCKS_CENTER);
    stop_movement();


    switch_state(1);

}

CY_ISR(ISR_Handler_7)
{
    UART_1_PutString("State 7: Detect colour 78\n");
            
    PWM_colour_Start();
    
    Counter_colour_left_Start();
    Counter_colour_right_Start();
    
    LED_colour_left_Write(1);   // Turn on LED
    LED_colour_right_Write(1);
    S0_left_Write(1);    // scaling frequency
    S0_right_Write(1);
    S1_left_Write(1);   
    S1_right_Write(1);
    CyDelay(50);
    
    compare_ready = 0;
    
    char str_colour[100];
    
    int red_freq_left = 0;
    int blue_freq_left = 0;
    int green_freq_left = 0;  
    int puck_colour_left = 0;
    
    int red_freq_right = 0;
    int blue_freq_right = 0;
    int green_freq_right = 0;
    int puck_colour_right = 0;
    
    for (int i = 0; i < 3; i++)
    {
        colour_sense(1, &red_freq_left, &red_freq_right);
        colour_sense(2, &blue_freq_left, &blue_freq_right);
        colour_sense(3, &green_freq_left, &green_freq_right);
    }
    
    if (valid_pucks[0] == 1){ close_servo(5); }
    if (valid_pucks[1] == 1){ close_servo(6); }
    
    puck_colour_left = colour_determine(red_freq_left/3,blue_freq_left/3,green_freq_left/3);
    puck_colour_right = colour_determine(red_freq_right/3,blue_freq_right/3,green_freq_right/3);
    
    valid_pucks[2] = (puck_colour_right == 3) ? 1 : 0;
    valid_pucks[3] = (puck_colour_left == 1) ? 1 : 0;
        
    sprintf(str_colour, "puck colour 8 (left) = %d (%d)\tpuck colour 7 (right) = %d (%d)\n", puck_colour_left, valid_pucks[3], puck_colour_right, valid_pucks[2]);
    UART_1_PutString(str_colour);
    
    LED_colour_left_Write(0);
    LED_colour_right_Write(0);
    
    PWM_colour_Stop();
    PWM_colour_Init();
    
    Counter_colour_left_Stop();    
    Counter_colour_left_Init();
    Counter_colour_right_Stop();
    Counter_colour_right_Init();
      
    switch_state(1);
     
}

CY_ISR(ISR_Handler_8)
{
    UART_1_PutString("State 8: Move backwards to center\n");
    
    move_straight(2);
    until_target_count(COUNT_DIST_PUCKS_CENTER);
    stop_movement();
    CyDelay(100);
    
    switch_state(1);

}

CY_ISR(ISR_Handler_9)
{
    UART_1_PutString("State 9: Flick valid pucks alternately\n");
    
    PWM_servo78_Start();
    
    if (valid_pucks[2] == 1){ 
        close_servo(7);
        flick(7); 
        init_servo(7);
        CyDelay(WAIT-3*ROD45);
    }
    if (valid_pucks[3] == 1){ 
        close_servo(8);
        flick(8); 
        init_servo(8);
        CyDelay(WAIT-3*ROD45);
    }
        
    int valid_id = 0;
    
    if (valid_pucks[3] == 1) { valid_id = 8; }
    else if (valid_pucks[2] == 1) { valid_id = 7; }
    else if (valid_pucks[1] == 1) { 
        valid_id = 6; 
        PWM_servo78_Stop();
        PWM_servo56_Start();
    }
    else if (valid_pucks[0] == 1) { 
        valid_id = 5; 
        PWM_servo78_Stop();
        PWM_servo56_Start();
    }
    
    close_servo(valid_id);
    for (;;){
        flick(valid_id);
        CyDelay(WAIT-ROD45);
    }
  
}

//********************end of SECTION 3: interrrupts function for switch and all the output states****************//



//********************start of SECTION 4: interrrupts function for activities of components****************//

//***STATE_3***//
CY_ISR(ISR_PWM_colour_Handler)
{
    PWM_colour_ReadStatusRegister();
    compare_ready = 1;

}

//********************end of SECTION 4: interrrupts function for activities of components****************//



//********************start of SECTION 5: main function****************//
int main(void)
{
    CyGlobalIntEnable; /* Enable global interrupts. */

    /* Place your initialization/startup code here (e.g. MyInst_Start()) */
    UART_1_Start();
    isr_1_StartEx(ISR_Handler_1);
    isr_2_StartEx(ISR_Handler_2);
    isr_3_StartEx(ISR_Handler_3);
    isr_4_StartEx(ISR_Handler_4);
    isr_5_StartEx(ISR_Handler_5);
    isr_6_StartEx(ISR_Handler_6);
    isr_7_StartEx(ISR_Handler_7);
    isr_8_StartEx(ISR_Handler_8);
    isr_9_StartEx(ISR_Handler_9);

    isr_SW1_StartEx(ISR_Handler_SW1);
    
    // State_3_Detect colour 56
    isr_PWM_colour_StartEx(ISR_PWM_colour_Handler);
    
    // State_4_Left_Gate
    PWM_servo56_Start();
    PWM_servo78_Start();
    
    init_servo(5);
    init_servo(6);
    init_servo(7);
    init_servo(8);
    
    PWM_servo56_Stop();
    PWM_servo78_Stop(); 
    
    for(;;)
    {
        /* Place your application code here. */
    }
}

//********************end of SECTION 5: main function****************//



//********************start of SECTION 6: other functions declaration****************//
// switch state
void switch_state(int n)
{
    if (input == 0){
        input = 1;
    }
    else{
        input = (input << n == 512) ? 1 : input << n;
    }
    
    char str_state[200];
    
    // control register = 0001
    Control_Reg_1_Write(input & 0x1f);
    Control_Reg_2_Write((input & 0x1e0) >> 5);
        
    sprintf(str_state, "input = %d (%d)\t control_reg_1 = %d (%d)\t control_reg_2 = %d (%d)\n", int_to_int(input), input, int_to_int(Control_Reg_1_Read()), Control_Reg_1_Read(), int_to_int(Control_Reg_2_Read()), Control_Reg_2_Read());
    UART_1_PutString(str_state);
    UART_1_PutString("In switching state\n");
    
    
             
    // input = 1  --> 00000 00001  --> state_2 
    // input = 2  --> 00000 00010  --> state_3
    // input = 4  --> 00000 00100  --> state_4
    // input = 8  --> 00000 01000  --> state_5
    // input = 16 --> 00000 10000  --> state_6
    // input = 32 --> 00001 00000  --> state_7
    // ...
}

unsigned int int_to_int(unsigned int k) {
    return (k == 0 || k == 1 ? k : ((k % 2) + 10 * int_to_int(k / 2)));
}

void flick(int id_valid_puck){
    char str_flick[100];
    if (id_valid_puck == 5){
        
        PWM_servo56_WriteCompare(FLICK_SERVO_PWM5);
        
        Control_Reg56_Write(1);
        sprintf(str_flick, "FLICK | servo5: %d\n", FLICK_SERVO_PWM5);
        UART_1_PutString(str_flick);
        
        CyDelay(ROD45);
        Control_Reg56_Write(0);
        
        CyDelay(2000);
        
        close_servo(5);
    }
    else if (id_valid_puck == 6){
        
        PWM_servo56_WriteCompare(FLICK_SERVO_PWM6);
        
        Control_Reg56_Write(2);        
        sprintf(str_flick, "FLICK | servo6: %d\n", FLICK_SERVO_PWM6);
        UART_1_PutString(str_flick);
       
        CyDelay(ROD45);
        Control_Reg56_Write(0);
        
        CyDelay(2000);
        
        close_servo(6);
    }
    else if (id_valid_puck == 7){
        
        PWM_servo78_WriteCompare(FLICK_SERVO_PWM7);
        
        Control_Reg78_Write(1);        
        sprintf(str_flick, "FLICK | servo7: %d\n", FLICK_SERVO_PWM7);
        UART_1_PutString(str_flick);
        
        CyDelay(ROD45);
        Control_Reg78_Write(0);
        
        CyDelay(2000);
        
        close_servo(7);
    }
    else if (id_valid_puck == 8){
        
        PWM_servo78_WriteCompare(FLICK_SERVO_PWM8);
        
        Control_Reg78_Write(2);        
        sprintf(str_flick, "FLICK | servo8: %d\n", FLICK_SERVO_PWM8);
        UART_1_PutString(str_flick);
        
        CyDelay(ROD45);
        Control_Reg78_Write(0);
        
        CyDelay(2000);
        
        close_servo(8);
    }
    
    
    
}

void init_servo(int id_puck){
    char str_init_servo[100];
    if (id_puck == 5){
        
        PWM_servo56_WriteCompare(INITI_SERVO_PWM5);
        
        Control_Reg56_Write(1);
        sprintf(str_init_servo, "INIT | servo5: %d\n", INITI_SERVO_PWM5);
        UART_1_PutString(str_init_servo);
        
        
    }
    else if (id_puck == 6){
        
        
        PWM_servo56_WriteCompare(INITI_SERVO_PWM6);
        Control_Reg56_Write(2);    
        
        sprintf(str_init_servo, "INIT | servo6: %d\n", INITI_SERVO_PWM6);
        UART_1_PutString(str_init_servo);

       
    }
    else if (id_puck == 7){
        
        PWM_servo78_WriteCompare(INITI_SERVO_PWM7);
        
        Control_Reg78_Write(1);                         
        sprintf(str_init_servo, "INIT | servo7: %d\n", INITI_SERVO_PWM7);
        UART_1_PutString(str_init_servo);

    }
    else if (id_puck == 8){
        
        PWM_servo78_WriteCompare(INITI_SERVO_PWM8);
        
        Control_Reg78_Write(2);                        
        sprintf(str_init_servo, "INIT | servo8: %d\n", INITI_SERVO_PWM8);
        UART_1_PutString(str_init_servo);

       
    }
    CyDelay(ROD45*2);
    Control_Reg56_Write(0);
    Control_Reg78_Write(0);
   
}

void close_servo(int id_puck){
    char str_close_servo[100];
    if (id_puck == 5){
        PWM_servo56_WriteCompare(CLOSE_SERVO_PWM5);
        
        Control_Reg56_Write(1);                    
        sprintf(str_close_servo, "CLOSE | servo5: %d\n", CLOSE_SERVO_PWM5);
        UART_1_PutString(str_close_servo);
    
       
    }
    else if (id_puck == 6){
        
        PWM_servo56_WriteCompare(CLOSE_SERVO_PWM6);
        
        Control_Reg56_Write(2);                        
        sprintf(str_close_servo, "CLOSE | servo6: %d\n", CLOSE_SERVO_PWM6);
        UART_1_PutString(str_close_servo);
 
    }
    else if (id_puck == 7){
        
        PWM_servo78_WriteCompare(CLOSE_SERVO_PWM7);
        
        Control_Reg78_Write(1);                        
        sprintf(str_close_servo, "CLOSE | servo7: %d\n", CLOSE_SERVO_PWM7);
        UART_1_PutString(str_close_servo);
       
    }
    else if (id_puck == 8){
        
        PWM_servo78_WriteCompare(CLOSE_SERVO_PWM8);
        
        Control_Reg78_Write(2);                        
        sprintf(str_close_servo, "CLOSE | servo8: %d\n", CLOSE_SERVO_PWM8);
        UART_1_PutString(str_close_servo);
       
    }
    CyDelay(ROD45);
    Control_Reg56_Write(0);
    Control_Reg78_Write(0);
}


//********************end of SECTION 6: other functions declaration****************//

/* [] END OF FILE */
