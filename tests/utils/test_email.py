import pytest

from q4l.utils.email_agent import EmailAgent


@pytest.fixture(scope="module")
def email_agent():
    return EmailAgent("your_email@gmail.com", "your_email_password")


def test_send_email_with_attachment(email_agent):
    receiver_email = "recipient_email@example.com"
    cc_list = ["cc_email1@example.com", "cc_email2@example.com"]
    subject = "Test Email with Attachments"
    body = "This is a test email with attachments"
    attachment_paths = ["tests/attachment1.pdf", "tests/attachment2.pdf"]

    email_agent.send_email(receiver_email, cc_list, subject, body, attachment_paths)

    # TODO: check if the email was sent successfully


def test_send_email_without_attachment(email_agent):
    receiver_email = "recipient_email@example.com"
    cc_list = ["cc_email1@example.com", "cc_email2@example.com"]
    subject = "Test Email without Attachments"
    body = "This is a test email without attachments"

    email_agent.send_email(receiver_email, cc_list, subject, body, [])

    # TODO: check if the email was sent successfully


def test_send_email_with_invalid_attachment_path(email_agent):
    receiver_email = "recipient_email@example.com"
    cc_list = ["cc_email1@example.com", "cc_email2@example.com"]
    subject = "Test Email with Invalid Attachment Path"
    body = "This is a test email with an invalid attachment path"
    attachment_paths = ["tests/invalid_attachment.pdf"]

    with pytest.raises(FileNotFoundError):
        email_agent.send_email(receiver_email, cc_list, subject, body, attachment_paths)
