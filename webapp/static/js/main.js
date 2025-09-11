// Main JavaScript file for Smart Legal Assistant
document.addEventListener('DOMContentLoaded', function() {
    console.log('Smart Legal Assistant initialized');

    // Attach submit handler to the question form
    const questionForm = document.getElementById('questionForm');
    if (questionForm) {
        questionForm.addEventListener('submit', handleQuestionSubmit);
    }
});

async function handleQuestionSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const questionData = Object.fromEntries(formData);

    // Get form elements
    const btn = form.querySelector('.ask-btn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.loading-spinner');
    const answerSection = document.getElementById('answerSection');
    const errorSection = document.getElementById('errorSection');

    // Show loading state
    setLoadingState(btn, btnText, spinner, true);
    hideSections(answerSection, errorSection);

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: questionData.question,
                top_k: parseInt(questionData.top_k) || 5,
                template: questionData.template || 'default'
            })
        });

        const result = await response.json();

        if (response.ok) {
            // Show answer
            renderAnswer(result);
            answerSection.classList.remove('d-none');
        } else {
            // Show error
            renderError(result);
            errorSection.classList.remove('d-none');
        }
    } catch (error) {
        console.error('Network error:', error);
        // Show network error
        renderError({ message: 'خطا در ارتباط با سرور. لطفاً مجدداً تلاش کنید.' });
        errorSection.classList.remove('d-none');
    } finally {
        // Reset button state
        setLoadingState(btn, btnText, spinner, false);
    }
}

function setLoadingState(btn, btnText, spinner, loading) {
    btn.disabled = loading;
    if (loading) {
        btnText.textContent = 'در حال پردازش...';
        spinner.style.display = 'inline-block';
    } else {
        btnText.textContent = '🔍 پرسش از دستیار';
        spinner.style.display = 'none';
    }
}

function hideSections(answerSection, errorSection) {
    answerSection.classList.add('d-none');
    errorSection.classList.add('d-none');
}

function renderAnswer(result) {
    // Render answer text (maintain paragraphs)
    const answerContent = document.getElementById('answerContent');
    if (answerContent) {
        answerContent.innerHTML = result.answer.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
        if (!answerContent.innerHTML.startsWith('<p>')) {
            answerContent.innerHTML = '<p>' + answerContent.innerHTML + '</p>';
        }
        if (!answerContent.innerHTML.endsWith('</p>')) {
            answerContent.innerHTML = answerContent.innerHTML.replace(/<\/p>$/, '') + '</p>';
        }
    }

    // Render citations with anchors
    const citationsContent = document.getElementById('citationsContent');
    if (citationsContent) {
        if (result.citations && result.citations.length > 0) {
            const citationsHtml = '<h6 class="text-secondary mb-2">📖 منابع:</h6><ul class="list-unstyled">' +
                result.citations.map((citation, index) => {
                    let citationText = `[${index + 1}] `;

                    if (citation.document_title) {
                        citationText += `<strong>${citation.document_title}</strong>`;
                    }

                    if (citation.article_number) {
                        citationText += ` - ماده ${citation.article_number}`;
                    }

                    if (citation.note_label) {
                        citationText += ` ${citation.note_label}`;
                    }

                    // Create anchor if link exists
                    if (citation.link) {
                        citationText += ` <a href="${citation.link}" target="_blank" rel="noopener noreferrer" class="text-decoration-none">[مشاهده]</a>`;
                    }

                    return `<li class="mb-1"><small>${citationText}</small></li>`;
                }).join('') + '</ul>';

            citationsContent.innerHTML = citationsHtml;
        } else {
            citationsContent.innerHTML = '';
        }
    }

    // Show meta info
    const metaInfo = document.getElementById('metaInfo');
    if (metaInfo && result.processing_time) {
        const chunks = result.retrieved_chunks || 0;
        const time = result.processing_time.toFixed(2);
        metaInfo.textContent = `${chunks} سند بازیابی شده | زمان پردازش: ${time} ثانیه`;
    }
}

function renderError(error) {
    const errorContent = document.getElementById('errorContent');
    if (errorContent) {
        let message = 'خطای نامشخص رخ داده است.';

        if (error.detail) {
            if (typeof error.detail === 'string') {
                message = error.detail;
            } else if (error.detail.message) {
                message = error.detail.message;
            }
        } else if (error.message) {
            message = error.message;
        }

        errorContent.textContent = message;
    }
}
