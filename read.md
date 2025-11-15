```mermaid
graph TD
    A[شروع: دریافت درخواست ارزیابی (EvaluationRequest)] --> B{آیا نتیجه در Cache موجود است؟};
    B -- بله --> C[بازگرداندن نتیجه از Cache];
    B -- خیر --> D[ایجاد Agent ها];
    D --> E{شروع Round ها (Rounds 1 to N)};
    E -- هر Round --> F[اجرای موازی یا ترتیبی Agent ها];
    F --> G[تجزیه و تحلیل نتایج Agent ها];
    G --> H{آیا شرایط Early Stopping برآورده شده است؟};
    H -- بله --> I[پایان Round ها];
    H -- خیر --> E;
    I --> J[تجمیع تحلیل‌های Agent ها];
    J --> K[برگزاری دادرسی هیئت منصفه (Jury Deliberation)];
    K --> L[کالیبراسیون و تولید حکم نهایی (Final Judgment)];
    L --> M[ایجاد نتیجه ارزیابی (EvaluationResult)];
    M --> N[ذخیره نتیجه در Cache];
    N --> O[پایان: بازگرداندن EvaluationResult];
    C --> O;
```