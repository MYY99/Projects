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

extern const int MASTER_SPEED;
extern const int KP;
extern const int COUNT_DIST_90DEGREES;

// general
void move_straight(int mode) // mode 1 = forward; mode 2 = backward
{
    // startup code
    PWM_R_Start(); 
    PWM_L_Start(); 
    QuadDec_R_Start();
    QuadDec_L_Start();

    // control direction of shaft
    int IN1 = 0;                    //***change IN1 to 0 and IN2 to 1 if opposite direction***
    int IN2 = 1;
    
    if (mode == 2)
    {
        IN1 = 1;
        IN2 = 0;
    }
    else if (mode != 1)
    {
        UART_1_PutString("ERROR! FORWARD = 1; BACKWARD = 2! NO OTHER INTEGERS FOR MODE!\n");
    }

    // control direction of DC motors (opp direction)
    Motor_R_IN_1_Write(IN1);
    Motor_R_IN_2_Write(IN2);
    Motor_L_IN_3_Write(IN1);
    Motor_L_IN_4_Write(IN2);

    
    // step 3: pwm of both wheels start with the same value
    PWM_R_WriteCompare(MASTER_SPEED);
    PWM_L_WriteCompare(MASTER_SPEED);
    
    // step 4: set encoder counts to 0 
    QuadDec_R_SetCounter(0);
    QuadDec_L_SetCounter(0);
}

void stop_movement()
{
    int a = 1;
    
    // stop both motors
    Motor_R_IN_1_Write(a);
    Motor_R_IN_2_Write(a);
    Motor_L_IN_3_Write(a);
    Motor_L_IN_4_Write(a);
    
    // stop PWM and quaddec
    PWM_R_Stop();
    PWM_L_Stop();
    QuadDec_R_Stop();
    QuadDec_L_Stop();
    
    PWM_R_Init();
    PWM_L_Init();
    QuadDec_R_Init();
    QuadDec_L_Init();
    /*QuadDec_R_SetCounter(0);
    QuadDec_L_SetCounter(0);*/

    UART_1_PutString("movement stopped\n");
}

// state 4
void turn (int mode)
{
    // should no change unless needed
    // initialization
    char string_1[100];
    int counter_R = 0;
    int counter_L = 0;
    int IN1 = 0;
    int IN2 = 0;
    
    if (mode == 1)
    {
        IN1 = 0;
        IN2 = 1;
    }
    else if (mode == 2)
    {
        IN1 = 1;
        IN2 = 0;
    }
    else 
    {
        UART_1_PutString("Error! 'turn' function receives mode 1 as left turn and mode 2 as right turn\n");
    }
    
    // startup code
    PWM_R_Start(); 
    PWM_L_Start(); 
    QuadDec_R_Start();
    QuadDec_L_Start();
    
    // parameters for change
    // PWM compare
    int speed_R = MASTER_SPEED;          //***(master)right wheel speed***
    int speed_L = MASTER_SPEED;         //***(slave) left wheel speed***

    // control direction of DC motors (opp direction)
    Motor_R_IN_1_Write(IN2);
    Motor_R_IN_2_Write(IN1);
    Motor_L_IN_3_Write(IN1);
    Motor_L_IN_4_Write(IN2);
    
    // step 3: pwm of both wheels start with the same value
    PWM_R_WriteCompare(speed_R);
    PWM_L_WriteCompare(speed_L);
    
    // step 4: set encoder counts to 0 
    QuadDec_R_SetCounter(0);
    QuadDec_L_SetCounter(0);
    
    // until both wheels reach count
    while (abs(counter_R) < COUNT_DIST_90DEGREES)
    {
        // shaft encoder count 
        counter_L = QuadDec_L_GetCounter();
        counter_R = QuadDec_R_GetCounter();
        
        // step 5: differential drive
        speed_L = ((abs(counter_R) - abs(counter_L))* KP) + speed_R;
        if (speed_L < 1)
        {
            speed_L = 100;
        }
        else if (speed_L > 24999)
        {
            speed_L = 24900;
        }

        PWM_L_WriteCompare(speed_L);
        
        // print the current count and new speed
        sprintf(string_1, "Right (Master): %d (%d)\nLeft (Slave): %d (%d)\n\n", counter_R, speed_R, counter_L, speed_L);
        UART_1_PutString(string_1);
        
        CyDelay(10);
    }
    
    stop_movement();

    UART_1_PutString("end of turn function\n");
}

// state 2
int until_ultrasonic_detected(int *dist_reach_flag)
{
    
    // should no change unless needed
    // initialization
    char string_1[80];
    int counter_R = 0;
    int counter_L = 0;
    
    // slave wheel PWM compare initialization
    int speed_L = MASTER_SPEED;         
    int for_print = 0;

    
    // until both wheels reach count
    while (*dist_reach_flag < 3)
    {
        // shaft encoder count 
        counter_L = QuadDec_L_GetCounter();
        counter_R = QuadDec_R_GetCounter();
        
        // step 5: differential drive
        speed_L = ((abs(counter_R) - abs(counter_L))* KP) + MASTER_SPEED;
        if (speed_L < 1)
        {
            speed_L = 100;
        }
        else if (speed_L > 24999)
        {
            speed_L = 24900;
        }
        PWM_L_WriteCompare(speed_L);
        
        for_print++;
        if (for_print >= 10)
        {
            // print the current count and new speed
            sprintf(string_1, "Right (Master): %d (%d)\nLeft (Slave): %d (%d)\n\n", counter_R, MASTER_SPEED, counter_L, speed_L);
            UART_1_PutString(string_1);
            
            for_print = 0;
        }
                
        CyDelay(10);
    }
    
    return abs(counter_R);
}

// state 3
void until_target_count(int target_count)
{
    
    // should no change unless needed
    // initialization
    char string_1[80];
    int counter_R = 0;
    int counter_L = 0;
    
    // slave wheel PWM compare initialization
    int speed_L = MASTER_SPEED;         
    int for_print = 0;
 
    // until both wheels reach count
    while (abs(counter_R) < target_count)
    {
        // shaft encoder count 
        counter_L = QuadDec_L_GetCounter();
        counter_R = QuadDec_R_GetCounter();
        
        // step 5: differential drive
        speed_L = ((abs(counter_R) - abs(counter_L))* KP) + MASTER_SPEED;
        if (speed_L < 1)
        {
            speed_L = 100;
        }
        else if (speed_L > 24999)
        {
            speed_L = 24900;
        }
        PWM_L_WriteCompare(speed_L);
        
        for_print++;
        if (for_print >= 10)
        {
            // print the current count and new speed
            sprintf(string_1, "Right (Master): %d (%d)\nLeft (Slave): %d (%d)\n\n", counter_R, MASTER_SPEED, counter_L, speed_L);
            UART_1_PutString(string_1);
            
            for_print = 0;
        }
                
        CyDelay(10);
    }
}

/* [] END OF FILE */
