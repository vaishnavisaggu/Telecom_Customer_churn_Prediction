function showToast(message)
{
    const toast =
        document.getElementById("toast");

    if (!toast)
        return;

    toast.textContent =
        message;

    toast.classList.add("show");

    setTimeout(
        function()
        {
            toast.classList.remove("show");
        },
        2200
    );
}