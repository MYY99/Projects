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

extern int compare_ready;

void colour_sense(int mode, int* freq_leftPtr, int* freq_rightPtr)
{
    int S2 = 0;
    int S3 = 0;
    char str_colour_sense[100];
    char colour[10];

    int count_left = 0;
    int count_right = 0;
    
    if (mode == 1) //red
    {
        S2 = 0;
        S3 = 0;
        strcpy(colour,"Red");
    }
    else if (mode == 2) //blue
    {
        S2 = 0;
        S3 = 1;
        strcpy(colour,"Blue");
    }
    else if (mode == 3) //green
    {
        S2 = 1;
        S3 = 1;
        strcpy(colour,"Green");
    }
    
    S2_left_Write(S2);
    S2_right_Write(S2);
    S3_left_Write(S3);
    S3_right_Write(S3);
    CyDelay(20);
    
    Control_Reg_colour_Write(1);
    CyDelay(1);
    Control_Reg_colour_Write(0);

    while (compare_ready == 0) {}

    count_left = Counter_colour_left_ReadCapture();
    count_right = Counter_colour_right_ReadCapture();
    sprintf(str_colour_sense, "left | %s: %d\nright | %s: %d\n\n", colour, count_left, colour, count_right);
    UART_1_PutString(str_colour_sense);

    compare_ready = 0;
    
    *freq_leftPtr += count_left;
    *freq_rightPtr += count_right;

}

int colour_determine(int red_freq, int blue_freq, int green_freq)
{
    if (red_freq > blue_freq && red_freq > green_freq)
    {
        return 1;
    }
    else if (blue_freq > (green_freq + 0.2*green_freq))
    {
        return 2;
    }
    else
    {
        return 3;
    }
}


/* [] END OF FILE */
