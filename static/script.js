// querySelector選不到特殊字元像是"-" 因此需加上兩個反斜線\\
const chatInput = document.querySelector(".chat\\-input textarea");
const sendChatBtn = document.querySelector(".chat\\-input span");
const chatbox = document.querySelector(".chatbot\\-content");
const chatToggler = document.querySelector(".chatbot\\-toggler");
const chatCloseBtn = document.querySelector(".close\\-btn");

let userMessage;

const createChatLi = (message, className) => {
    // create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", className);
    let chatContent =
        className === "outgoing"
            ? `<p></p>`
            : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
};


document.addEventListener("DOMContentLoaded", function () {
    const sendBtn = document.getElementById("send-btn");
    const textarea = document.querySelector(".chat-input textarea");

    let enterPressedOnce = false; // 追踪是否已經按下了一次Enter键
    let waitingForResponse = false; // 追踪是否正在等待服務器響應(當用戶送出一則訊息後，系統未回覆之前不得送出下一筆)
    let pasted = false; // 追踪訊息是否用了複製貼上，而不是手動輸入

    textarea.addEventListener('input', function (event) {
        if (event.inputType === "insertFromPaste") {
            pasted = true;
        }
    });

    // 當用戶在對話筐按下鍵盤時
    textarea.addEventListener('keydown', function (event) {
        // 檢查是否按下了 Enter
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();

             if ((enterPressedOnce || pasted) && !waitingForResponse) {
                sendBtn.click();
                enterPressedOnce = false;
                pasted = false; // 重置貼上狀態
                textarea.value = "";
            } else if (!pasted) {
                enterPressedOnce = true;
            }
        } else if (event.key !== 'Enter') {
            enterPressedOnce = false;
        }
    });

    sendBtn.addEventListener("click", function () {
        const message = textarea.value;
        console.log(message)
        if (message.trim() !== "" && !waitingForResponse) {
            // 設置等待狀態
            waitingForResponse = true;

            // 使用 AJAX 送訊息到 Flask
            fetch("/send_message", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: message,
                }),
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.error) {
                        console.error("Failed to send message.");
                    } else {
                        // 清空 textarea
                        // textarea.value = "";

                        // 把回傳的訊息傳到聊天室
                        const chatContent = document.querySelector(".chatbot-content");
                        const responseLi = document.createElement("li");
                        responseLi.className = "chat incoming";

                        const responseSpan = document.createElement("span");
                        responseSpan.className = "material-symbols-outlined";
                        responseSpan.textContent = "smart_toy";
                        responseLi.appendChild(responseSpan);

                        const responseP = document.createElement("p");
                        responseP.textContent = data.response_message;  // 使用server回應的訊息
                        responseLi.appendChild(responseP);

                        chatContent.appendChild(responseLi);

                        // 自動滾動到聊天室底部
                        chatbox.scrollTo(0, chatbox.scrollHeight);
                    }
                    // 重置等待狀態
                    waitingForResponse = false;
                });
        }
    });
});


// 使用者輸入的訊息
const handleChat = () => {
    userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // append the user's message to the chatbox
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    // auto scroll to the bottom if chat is overflow
    chatbox.scrollTo(0, chatbox.scrollHeight);

    setTimeout(() => {
        // Display "Thinking..." message while waiting for the Response
        // const incomingChatLi = createChatLi("Thinking", "incoming");
        // chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        // generateResponse(incomingChatLi);
    }, 600);
};


sendChatBtn.addEventListener("click", handleChat);
chatToggler.addEventListener("click", () =>
    document.body.classList.toggle("show-chatbot")
);
chatCloseBtn.addEventListener("click", () =>
    document.body.classList.remove("show-chatbot")
);


// 檢查是否顯示發送按鈕
textarea.addEventListener('input', function () {
    if (textarea.value.trim() !== '') {
        sendBtn.style.display = 'inline-block';
    } else {
        sendBtn.style.display = 'none';
    }
});

